import torch
import typing
import numpy as np
from modified_env import VIEWPORT_H, VIEWPORT_W, SCALE, FPS, LEG_DOWN
from dataclasses import dataclass

import gymnasium as gym

import os

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class Trajectory:
    '''
        Wrapper for trajectory data
    '''
    
    def __init__(self, obs = None, actions = None, rewards = None, dones = None, logprobs=None, values=None):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.logprobs = logprobs
        self.values = values
        
        self.num_steps, self.num_envs = self.obs.shape[:2]
        
    def set_default(self, envs, num_steps, num_envs, device):
        self.obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        
    
    def get_values(self, agent):
        self.values = torch.zeros_like(self.rewards)
        for step in self.num_steps:
            self.values[step] = agent.get_values(self.obs[step]).reshape(1, -1)
        return self.values
        

def adv_calc(agent, trajectory: Trajectory, gamma, gae_lambda, device='cpu', refill=True):
    '''
    Calculates a single trajectory estimate of PDL Value difference
    Uses GAE lambda
    '''
    if refill or trajectory.values is None:
        trajectory.get_values(agent)
    with torch.no_grad():
        next_value = torch.zeros(1, trajectory.num_envs)
        advantages = torch.zeros(trajectory.num_steps, trajectory.num_envs).to(device)
        next_done = torch.ones(trajectory.num_envs)
        lastgaelam = 0
        for t in reversed(range(trajectory.num_steps)):
            if t == trajectory.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - trajectory.dones[t + 1]
                nextvalues = trajectory.values[t + 1]
            delta = trajectory.rewards[t] + gamma * nextvalues * nextnonterminal - trajectory.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

class EnsembleDist:
    '''
        Returns samples from a joint distribution
        Used to sample params for reward shaping
    '''
    
    def __next__(self):
        raise NotImplementedError
            
class UniformProductDist(EnsembleDist):
    '''
        Returns uniformly distributed samples over a hypercuboid
    '''
    
    def __init__(self, num_params, param_min, param_size):
        self.num_params = num_params
        self.param_min = param_min
        self.param_size = param_size
        
    def __next__(self):
        rands = torch.rand(self.num_params)
        return rands * self.param_size + self.param_min
    
class FixedDist(EnsembleDist):
    
    def __init__(self, fixed_params):
        self.fixed_params = fixed_params
        
    def __next__(self):
        return self.fixed_params
    
# Abstract class that implements the reward shaping
# Designed to be called after the step function
class RewardShaper:
    
    def get_shaped_reward(self, state=None, action=None, aux=None):
        raise NotImplementedError
    
class MultiRewardShaper(RewardShaper):  
    '''Takes params and returns reward absed on state, action, env hidden state, agent''' 
    
    def __init__(self, vector_envs, agent=None):
        self.vector_envs = vector_envs
        self.agent = agent
        
        self.num_envs = len(vector_envs.envs)
        self.prev_rewards = [None for _ in range(self.num_envs)]
        
    def register(self, agent):
        self.agent = agent
        
    def get_shaped_reward(self, state, action, aux):
        raise NotImplementedError
    
class MultiLunarShaper(MultiRewardShaper):
    
    def __init__(self, vector_envs, agent=None):
        super().__init__(vector_envs, agent)
        self.dim = 5
        
    def apply_state_func(self, state_funcs):
        for idx, env in enumerate(self.vector_envs.envs):
            env.unwrapped.state = state_funcs[idx](env.unwrapped.state)
        
    
    def get_shaped_reward(self, prev_state, action, aux):
        # Aux is a dict
        if aux['params'] is None:
            return np.zeros_like(aux['env_reward']), aux['env_reward']
        shaped_rewards, task_rewards = [], []
        for idx, env in enumerate(self.vector_envs.envs):
            shaped_reward, task_reward = 0, 0
            
            prev_env_state = prev_state[idx]
            env_action = action[idx]
            
            env_helper = env.unwrapped
            m_power = 0.0
            if (env_helper.continuous and env_action[0] > 0.0) or (
                not env_helper.continuous and env_action == 2
            ):
                # Main engine
                if env_helper.continuous:
                    m_power = (np.clip(env_action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                    assert m_power >= 0.5 and m_power <= 1.0
                else:
                    m_power = 1.0

            s_power = 0.0
            if (env_helper.continuous and np.abs(env_action[1]) > 0.5) or (
                not env_helper.continuous and env_action in [1, 3]
            ):
                # Orientation engines
                if env_helper.continuous:
                    s_power = np.clip(np.abs(env_action[1]), 0.5, 1.0)
                    assert s_power >= 0.5 and s_power <= 1.0
                else:
                    s_power = 1.0
            
            pos = env_helper.lander.position
            vel = env_helper.lander.linearVelocity
            
            env_state = [
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (env_helper.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                env_helper.lander.angle,
                20.0 * env_helper.lander.angularVelocity / FPS,
                1.0 if env_helper.legs[0].ground_contact else 0.0,
                1.0 if env_helper.legs[1].ground_contact else 0.0,
            ]
            shaping = (
                - aux['params'][idx][0] * np.sqrt(env_state[0] * env_state[0] + env_state[1] * env_state[1])
                - aux['params'][idx][1] * np.sqrt(env_state[2] * env_state[2] + env_state[3] * env_state[3])
                - aux['params'][idx][2] * abs(env_state[4])
                + aux['params'][idx][3] * env_state[6]
                + aux['params'][idx][4] * env_state[7]
            )  # And ten points for legs contact, the idea is if you
            # lose contact again after landing, you get negative reward
            if self.prev_rewards[idx] is not None:
                shaped_reward = shaping - self.prev_rewards[idx]
            self.prev_rewards[idx] = shaping

            shaped_reward -= (
                m_power * 0.30
            )  # less fuel spent is better, about -30 for heuristic landing
            shaped_reward -= s_power * 0.03

            if aux['env_reward'][idx] == -100 or aux['env_reward'][idx] == +100:
                task_reward = aux['env_reward'][idx]
            shaped_rewards.append(shaped_reward)
            task_rewards.append(task_reward)
        return np.array(shaped_rewards), np.array(task_rewards)
    
    def compute_next_step(self, prev_state, action, aux):
        shaped_rewards, task_rewards = [], []
        for idx, env in enumerate(self.vector_envs.envs):
            shaped_reward, task_reward = 0, 0
            
            prev_env_state = prev_state[idx]
            env_action = action[idx]
            
            env_helper = env.unwrapped
            m_power = 0.0
            if (env_helper.continuous and env_action[0] > 0.0) or (
                not env_helper.continuous and env_action == 2
            ):
                if env_helper.continuous:
                    m_power = (np.clip(env_action[0], 0.0, 1.0) + 1.0) * 0.5
                    assert m_power >= 0.5 and m_power <= 1.0
                else:
                    m_power = 1.0

            s_power = 0.0
            if (env_helper.continuous and np.abs(env_action[1]) > 0.5) or (
                not env_helper.continuous and env_action in [1, 3]
            ):
                if env_helper.continuous:
                    s_power = np.clip(np.abs(env_action[1]), 0.5, 1.0)
                    assert s_power >= 0.5 and s_power <= 1.0
                else:
                    s_power = 1.0
            
            pos = env_helper.lander.position
            vel = env_helper.lander.linearVelocity
            
            env_state = [
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (env_helper.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                env_helper.lander.angle,
                20.0 * env_helper.lander.angularVelocity / FPS,
                1.0 if env_helper.legs[0].ground_contact else 0.0,
                1.0 if env_helper.legs[1].ground_contact else 0.0,
            ]
            shaping = (
                - aux['params'][idx][0] * np.sqrt(env_state[0] * env_state[0] + env_state[1] * env_state[1])
                - aux['params'][idx][1] * np.sqrt(env_state[2] * env_state[2] + env_state[3] * env_state[3])
                - aux['params'][idx][2] * abs(env_state[4])
                + aux['params'][idx][3] * env_state[6]
                + aux['params'][idx][4] * env_state[7]
            )
            if self.prev_rewards[idx] is not None:
                shaped_reward = shaping - self.prev_rewards[idx]

            shaped_reward -= m_power * 0.30
            shaped_reward -= s_power * 0.03

            if aux['env_reward'][idx] == -100 or aux['env_reward'][idx] == +100:
                task_reward = aux['env_reward'][idx]
            shaped_rewards.append(shaped_reward)
            task_rewards.append(task_reward)
        return np.array(shaped_rewards), np.array(task_rewards)
    
    
    
def MH_sampler(seed_func, num_samples, ratio_func, prop_func, burn_in = 10, reject_handler=None):
    states = torch.stack([seed_func() for _ in range(num_samples)], dim=0)
    for _ in range(burn_in):
        proposals = prop_func(states)
        if reject_handler is not None:
            reject_handler.register(states)
        acc_prob = torch.min(1, ratio_func(proposals, states) / prop_func(proposals, states))
        rand_prob = torch.rand(states.shape)
        mask = (rand_prob < acc_prob).int()
        states = torch.where(mask, proposals, states)
        if reject_handler is not None:
            reject_handler.restore(mask)
    return states

def param_seed(low, high):
    num_params = low.shape
    return torch.rand((num_params,)) * (high - low) + low