import gymnasium as gym
import numpy as np

from ppo import make_env
from random.helpers import MultiLunarShaper

run_name = "0"
capture_video = False
env_id = "LunarLander-v2"
num_envs = 4


vec_env = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )

action_space = vec_env.action_space
lunar_shaper = MultiLunarShaper(vec_env)

num_steps = 100
seed = 1
next_obs, _ = vec_env.reset(seed=seed)
aux = {'params': [np.array([100., 100., 100., 10., 10.]) for _ in range(num_envs)]}
for step in range(0, num_steps):
    # TRY NOT TO MODIFY: execute the game and log data.
    action = action_space.sample()
    print(f"Running step: {step}")
    next_obs, env_reward, terminations, truncations, infos = vec_env.step(action)
    aux['env_reward'] = env_reward
    shaped_reward, task_reward = lunar_shaper.get_shaped_reward(next_obs, action, aux)
    print(f"Shaped Reward: {shaped_reward}")
    print(f"Task Reward  : {task_reward}")
    print(f"Env Reward   : {env_reward}")
    next_done = np.logical_or(terminations, truncations)
    if next_done.any():
        break