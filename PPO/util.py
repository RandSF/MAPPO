import numpy as np
import torch
import torch.nn as nn

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
    
    
def init(layer):
    nn.init.orthogonal(layer.weight.data,gain=1)
    nn.init.normal(layer.bias.data,std=0.01)
    return layer

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def t2n(t):
    return t.detach().cup().numpy()