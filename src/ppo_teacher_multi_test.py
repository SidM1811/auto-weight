# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from agents import Agent, ParamConditionedAgent, StateGenerator

import json

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

    env_id: str = "LunarLander-v3"
    """the id of the environment"""
    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_teacher_envs: int = 64
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
    num_shaping: int = 6
    """the number of shaping functions"""
    save_step: int = 25_000
    """save model every n steps"""
    
gym.register("LunarLanderCustom", entry_point="lunarlander_mod:LunarLander")

def make_env_id(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def sample_teacher_params(num_teacher_envs):
    random_weights = torch.rand(num_teacher_envs, args.num_shaping) * 2.0
    default_shaping = torch.tensor([100.0, 100.0, 100.0, 10.0, 0.3, 0.03]).repeat(num_teacher_envs, 1)
    return default_shaping * random_weights
    # return torch.zeros(num_teacher_envs, args.num_shaping)

if __name__ == "__main__":
    args_json = json.load(open('./combined_config.json',))
    # args = tyro.cli(Args)
    args = Args(**args_json)
    args.num_envs = args.num_teacher_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    last_save = 0
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(f"models/{run_name}"):
        os.makedirs(f"models/{run_name}")
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env_id(args.env_id, i, False, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    teacher_agent = ParamConditionedAgent(envs, args.num_shaping).to(device)
    teacher_optimizer = optim.Adam(teacher_agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    shaped_rewards = torch.zeros((args.num_steps, args.num_teacher_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    shaped_values = torch.zeros((args.num_steps, args.num_teacher_envs)).to(device)
    params = torch.zeros((args.num_steps, args.num_teacher_envs, args.num_shaping)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    teacher_params = sample_teacher_params(args.num_teacher_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Teacher params fixed for the iteration
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            params[step] = teacher_params

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_ext, value_int = teacher_agent.get_action_and_value(next_obs, params = teacher_params)
                
            value = value_ext
            shaped_value = value_int
            
            values[step] = value.flatten()
            shaped_values[step] = shaped_value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            current_obs = next_obs.clone()
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            shaped_reward = np.array([
                env.unwrapped.calculate_shaping_reward(current_obs[idx], next_obs[idx], action, shaping_weights=teacher_params[idx])
                for idx, (env, action) in enumerate(zip(envs.envs, action))
            ])
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            shaped_rewards[step] = torch.tensor(shaped_reward).to(device).view(-1)
            
            done_idx = np.where(next_done)[0]
            num_done = len(done_idx)
            teacher_params[done_idx] = sample_teacher_params(num_done).to(device)
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # add shaped reward to teacher reward - termination handling
            shaped_rewards[step] = torch.where(
                next_done > 0.0, 
                shaped_rewards[step], 
                torch.tensor(0.0).to(device))
            
            if infos and "episode" in infos:
                for idx, finished in enumerate(infos["_episode"]):
                    if finished:
                        print(f"global_step={global_step}, episodic_return={infos['episode']['r'][idx]}, episodic_length={infos['episode']['l'][idx]}")
                        writer.add_scalar(f"charts/episodic_return", infos["episode"]["r"][idx], global_step)
                        writer.add_scalar(f"charts/episodic_length", infos["episode"]["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_ext_value, next_int_value = teacher_agent.get_value(next_obs, params = teacher_params)
            next_ext_value = next_ext_value.reshape(1, -1)
            next_int_value = next_int_value.reshape(1, -1)
            
            next_value = next_ext_value.flatten()
            next_shaped_value = next_int_value.flatten()
            
            advantages = torch.zeros_like(rewards).to(device)
            shaped_advantages = torch.zeros_like(shaped_rewards).to(device)
            
            lastgaelam = 0
            lastshapedgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    nextshapedvalues = next_shaped_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    nextshapedvalues = shaped_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                shaped_delta = shaped_rewards[t] + args.gamma * nextshapedvalues * nextnonterminal - shaped_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                shaped_advantages[t] = lastshapedgaelam = shaped_delta + args.gamma * args.gae_lambda * nextnonterminal * lastshapedgaelam
            returns = advantages + values
            shaped_returns = shaped_advantages + shaped_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        
        b_advantages = advantages.reshape(-1)
        b_shaped_advantages = shaped_advantages.reshape(-1)
        
        b_returns = returns.reshape(-1)
        b_shaped_returns = shaped_returns.reshape(-1)
        
        b_values = values.reshape(-1)
        b_shaped_values = shaped_values.reshape(-1)
        
        b_params = params.reshape(-1, args.num_shaping)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newextvalue, newintvalue = teacher_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], params = b_params[mb_inds])
                newvalue = newextvalue
                newshapedvalue = newintvalue
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                newshapedvalue = newshapedvalue.view(-1)
                
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                    
                    # Shaped value loss
                    shaped_v_loss_unclipped = (newshapedvalue - b_shaped_returns[mb_inds]) ** 2
                    shaped_v_clipped = b_shaped_values[mb_inds] + torch.clamp(
                        newshapedvalue - b_shaped_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    shaped_v_loss_clipped = (shaped_v_clipped - b_shaped_returns[mb_inds]) ** 2
                    shaped_v_loss_max = torch.max(shaped_v_loss_unclipped, shaped_v_loss_clipped)
                    shaped_v_loss = 0.5 * shaped_v_loss_max.mean()
                    v_loss += shaped_v_loss
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    v_loss += 0.5 * ((newshapedvalue - shaped_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                teacher_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(teacher_agent.parameters(), args.max_grad_norm)
                teacher_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", teacher_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if global_step - last_save > args.save_step or global_step >= args.total_timesteps:
            last_save = global_step
            torch.save(teacher_agent.state_dict(), f"models/{run_name}/teacher_agent_{iteration}.pth")

    envs.close()
    writer.close()