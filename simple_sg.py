from collections import namedtuple
import numpy as np

from onpolicy.envs.mpe.core import Agent, Landmark, World
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f'{base_name}_{base_index}'
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.5, 0.5, 0.5])
            # landmark.color[i + 1] += 0.8
            landmark.index = i
        # set goal landmark
        goal = np_random.choice(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world:World):
        # the agents' negtive distance to the goal and the adversaries' positive distance to the goal
        reward=0.
        for a in world.agents:
            if a.adversary:
                reward+=np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            else:
                reward-=np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
        return reward

    def adversary_reward(self, agent, world):
        # the agents' positive distance to the goal and the adversaries' negtive distance to the goal
        reward=0.
        for a in world.agents:
            if a.adversary:
                reward-=np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            else:
                reward+=np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
        return reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # every agent observes itself's velocity, all the entities' color and relative position 
        
        landmarks_pos = []
        others_pos = []
        landmarks_color=[]
        others_color=[]
        comm = [] # communication of all other agents
        for entity in world.landmarks:  # world.entities:
            landmarks_pos.append(entity.state.p_pos - agent.state.p_pos)
            landmarks_color.append(entity.color) # entity colors
            
        for other in world.agents:
            if other is agent:continue
            others_pos.append(other.state.p_pos - agent.state.p_pos)
            others_color.append(other.color)
            comm.append(other.state.c)
        return np.concatenate( agent.state.p_vel + agent.state.p_pos + landmarks_pos  + others_pos)    
        # Observation=namedtuple("Observation",["vel","goal_pos","color","landmark_pos","landmark_color","other_pos","other_color"])
        # return Observation(vel = agent.state.p_vel, goal_pos = agent.goal_a.state.p_pos-agent.state.p_pos,
        #                    color = agent.color, landmark_pos = landmarks_pos, landmark_color = landmarks_color,
        #                    other_pos = others_pos, other_color = others_color)
