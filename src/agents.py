import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StateGenerator(nn.Module):
    
    def __init__(self, state_dim, param_dim):
        super(StateGenerator, self).__init__()
        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(state_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
        )
        self.param_encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(param_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
        )
        
        self.state_proj = nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.param_proj = nn.Sequential(
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
    def state_ratio(self, state):
        encoded_state = self.state_encoder(state)
        return self.state_proj(encoded_state)
    
    def param_ratio(self, param, state):
        encoded_param = self.param_encoder(param)
        encoded_state = self.state_encoder(state)
        return self.param_proj(encoded_param * encoded_state)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
class ParamConditionedAgent(nn.Module):
    
    def __init__(self, envs, param_dim):
        super().__init__()
        # Captures external value function
        self.ext_critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Captures internal value function
        self.int_critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Policy
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.param_critic = nn.Linear(param_dim, 1)
        self.param_actor = nn.Linear(param_dim, envs.single_action_space.n)
        self.params = None

    def get_value(self, x, params=None):
        # param normalization is important
        if params is None:
            if self.params is None:
                raise ValueError("Params not set")
            params = self.params
        param_adj = self.param_critic(params)
        return self.ext_critic(x), self.int_critic(x) * (1. + param_adj)

    def get_action_and_value(self, x, action=None, params=None):
        # param normalization is important
        if params is None:
            if self.params is None:
                raise ValueError("Params not set")
            params = self.params
        param_adj = self.param_actor(params)
        base_logits = self.actor(x)
        #  Single obs, action multi-params
        if len(param_adj.shape) < len(base_logits.shape):
            param_adj = param_adj.unsqueeze(-1)
        logits = base_logits * (1. + param_adj)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.ext_critic(x), self.int_critic(x) * (1. + self.param_critic(params))