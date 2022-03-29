from lib2to3.pytree import convert
from tkinter.messagebox import NO
from turtle import forward
from matplotlib.style import available
import numpy as np
from sympy import N
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .distributes import FixedNormal
from .popart import PopArt
from .util import check, init


# return a DiagGuassian distribution (the covariance of variants is 0), which is the base to get the action
class GuassianLayer(nn.Module): # with orthogonal
    def __init__(self,in_dim,out_dim): # output includes action distribution's mean and std
        super(GuassianLayer,self).__init__()
        self.mean_layer=nn.Linear(in_dim,out_dim)
        self.logstd_layer=nn.parameter(torch.zeros(out_dim).unsqueeze(1))
        init(self.mean_layer)
    
    def forward(self,x):
        action_mean=self.action_layer(x)
        
        # why it is an ugly implementation of *KFAC*
        # it looks like a straightward estimation of std via the weight of logstd_layer
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        
        if zeros.dim() == 2:
            bias = self.logstd_layer.t().view(1, -1)
        else:
            bias = self.logstd_layer.t().view(1, -1, 1, 1)
        action_logstd=zeros+bias
        
        return FixedNormal(action_mean,action_logstd.exp())
    
"""    
args={
    use_feature_normalization
    layer_num
    hidden_size
    recurrent_N
    *use_popart
    *use-orthogonal
}   
""" 

class ActLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ActLayer,self).__init__()
        self.action_out=GuassianLayer(in_dim,out_dim)
        
    def forward(self, x, available_actions=None, deterministic=False):
        action_logits=self.action_out(x,available_actions)
        actions=action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs=action_logits.log_probs(actions)
        return actions, action_log_probs
    
    def get_probs(self, x, available_actions=None):
        action_logits=self.action_out(x, available_actions)
        action_probs=action_logits.prob
        return action_probs
    
    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy
class MLPLayer(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
                #  in_dim,hidden_size,layer_num=2,):
        super(MLPLayer,self).__init__()
        self._layer_num=args.layer_num
        in_dim=obs_shape[0]
        hidden_size=args.hidden_size
        layer_num=args.layer_num
        
        self.norm=nn.LayerNorm(in_dim) if args.use_feature_normalization else None
        self.fc1=nn.Sequential(
            init(nn.Linear(in_dim,hidden_size)), F.relu, nn.LayerNorm(hidden_size))
        self.fc2=[]
        for _ in range(layer_num):
            self.fc2.append(copy.deepcopy(nn.Sequential(
                                               init(nn.Linear(hidden_size,hidden_size)), F.relu, nn.LayerNorm(hidden_size))))
    
    def forward(self,x):
        if self.norm is not None:
            x=self.norm(x)
        x=self.fc1(x)
        for i in range(self._layer_num):
            x=self.fc2[i](x)
        return x
    
class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N,):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
                # else:
                #     nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
    
class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor,self).__init__()
        
        # self._gain = args.gain
        # self._use_orthogonal = args.use_orthogonal
        # self._use_policy_active_masks = args.use_policy_active_masks
        # self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        # self._use_recurrent_policy = args.use_recurrent_policy
        # self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.base = MLPLayer(args, obs_space.shape) 
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.act = ActLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)
        
    def forward(self, obs, rnn_states, available_actions=None, deterministic=False):
        obs=check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        # if active_masks is not None:
        #     active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states)
        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, rnn_states, action,available_actions=None):
        obs=check(obs).to(**self.tpdv)
        rnn_states=check(rnn_states).to(**self.tpdv)
        action=check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions=check(available_actions).to(**self.tpdv)
        
        action_features=self.base(obs)
        
        action_features, rnn_states=self.rnn(action_features,rnn_states)
        
        action_log_probs, dist_entropy=self.act.evaluate_actions(action_features,action,available_actions)
        
        return action_log_probs, dist_entropy
    
class Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic,self).__init__()
        self._hidden_size = args.hidden_size
        # self._use_orthogonal = args.use_orthogonal
        # self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        # self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        # init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        
        self.base=MLPLayer(args, cent_obs_space)
        self.rnn=RNNLayer(self._hidden_size,self._hidden_size,self._recurrent_N)
        
        self.v_out=init(PopArt(self._hidden_size,1,device=device))
        
        self.to(device)
        
        def forward(cent_obs,rnn_states):
            cent_obs=check(cent_obs).to(**self.tpdv)
            rnn_states=check(rnn_states).to(**self.tpdv)
            # masks = check(masks).to(**self.tpdv)

            critic_features = self.base(cent_obs)
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)
            values = self.v_out(critic_features)

            return values, rnn_states