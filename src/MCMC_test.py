from MCMC import MCMC
from lunarlander_mod import LunarLander
import agents

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

num_chains = 4
num_samples = 10

envs = gym.vector.SyncVectorEnv([lambda: LunarLander() for _ in range(num_chains)])
base_agent = agents.Agent(envs)
_, _ = envs.reset()

class IWeight(nn.Module):
    
    def __init__(self, envs):
        super(IWeight, self).__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class StateManager():
    
    def __init__(self, state, actor_model, prob_model, internal_readout, external_readout):
        self.state = state
        self.prob_model = prob_model
        self.actor_model = actor_model
        self.internal_readout = lambda : internal_readout(self.state)
        self.external_readout = lambda : external_readout(self.state)
        self.prev_int = None
        self.cur_int = None
        self.prev_ext = None
        self.cur_ext = None
        self.accept_prob = None
        
    def propose(self):
        self.prev_int = self.internal_readout()
        self.prev_ext = self.external_readout()
        act_logits = self.actor_model(self.prev_ext)
        actions = torch.distributions.Categorical(logits=act_logits).sample()
        self.state.step(actions.numpy())
        self.cur_int = self.internal_readout()
        self.cur_ext = self.external_readout()
        
    def acceptance_probability(self):
        log_prob_diff = self.prob_model(self.cur_ext) - self.prob_model(self.prev_ext)
        self.accept_prob = torch.minimum(torch.tensor(1.0), torch.exp(log_prob_diff))
        
    def commit(self):
        mask = torch.rand(self.accept_prob.shape) < self.accept_prob
        for idx, env in enumerate(self.state.envs):
            if not mask[idx]:
                env.set_internal_state(self.prev_int[idx])
                
    def summary(self):
        return self.external_readout()
    
internal_readout = lambda envs: [env.readout_internal_state() for env in envs.envs]
external_readout = lambda envs: torch.Tensor([env.readout_external_state() for env in envs.envs])
state_manager = StateManager(envs, base_agent.actor, IWeight(envs), internal_readout, external_readout)    
mcmc_helper = MCMC(
    state_manager,
    num_samples=num_samples, 
    num_chains=num_chains
)

def mcmc_test():
    samples = mcmc_helper.sample()
    print(samples)
    
if __name__ == "__main__":
    mcmc_test()