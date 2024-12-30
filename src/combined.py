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
    num_student_envs: int = 4
    num_teacher_envs: int = 4
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
    il_coef: float = 0.0
    """coefficient of the il loss"""
    is_coef: float = 0.0
    """coefficient of the is loss"""
    kl_imp_coef: float = 0.0
    """coefficient of the kl importance sampling"""
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
    random_weights = torch.rand(num_teacher_envs, 6) * 2.0
    default_shaping = torch.tensor([100.0, 100.0, 100.0, 10.0, 0.3, 0.03]).repeat(num_teacher_envs, 1)
    return default_shaping * random_weights

if __name__ == "__main__":
    args_json = json.load(open('./combined_config.json',))
    # args = tyro.cli(Args)
    args = Args(**args_json)
    args.num_envs = args.num_student_envs + args.num_teacher_envs
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

    student_agent = Agent(envs).to(device)
    teacher_agent = ParamConditionedAgent(envs, args.num_shaping).to(device)
    student_optimizer = optim.Adam(student_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    teacher_optimizer = optim.Adam(teacher_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    state_generator = StateGenerator(envs.single_observation_space.shape[0], args.num_shaping).to(device)

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

    for iteration in range(1, args.num_iterations + 1):
        # Teacher params fixed for the iteration
        teacher_params = sample_teacher_params(args.num_teacher_envs).to(device)
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            student_optimizer.param_groups[0]["lr"] = lrnow
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            params[step] = teacher_params

            # ALGO LOGIC: action logic
            with torch.no_grad():
                
                action_student, logprob_student, _, value_student = student_agent.get_action_and_value(next_obs[:args.num_student_envs])
                action_teacher, logprob_teacher, _, value_ext_teacher, value_int_teacher = teacher_agent.get_action_and_value(next_obs[args.num_student_envs:], params = teacher_params)
                
            value_teacher = value_ext_teacher + value_int_teacher
            action = torch.cat([action_student, action_teacher])
            logprob = torch.cat([logprob_student, logprob_teacher])
            value = torch.cat([value_student, value_teacher])
            shaped_value = value_int_teacher
            
            values[step] = value.flatten()
            shaped_values[step] = shaped_value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            current_obs = next_obs.clone()
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            shaped_reward = np.zeros(args.num_teacher_envs)
            # shaped_reward = np.array([
            #     env.unwrapped.calculate_shaping_reward(current_obs[idx + args.num_student_envs], next_obs[idx + args.num_student_envs], action, shaping_weights=teacher_params[idx])
            #     for idx, (env, action) in enumerate(zip(envs.envs[args.num_student_envs:], action[args.num_student_envs:]))
            # ])
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            shaped_rewards[step] = torch.tensor(shaped_reward).to(device).view(-1)
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # add shaped reward to teacher reward - termination handling
            rewards[step][args.num_student_envs:] += torch.where(
                next_done[args.num_student_envs:] > 0.0, 
                shaped_rewards[step], 
                torch.tensor(0.0).to(device))
            
            if infos and "episode" in infos:
                for idx, finished in enumerate(infos["_episode"]):
                    ts_prefix = "teacher" if idx >= args.num_student_envs else "student"
                    if finished:
                        print(f"global_step={global_step}, episodic_{ts_prefix}_return={infos['episode']['r'][idx]}, episodic_{ts_prefix}_length={infos['episode']['l'][idx]}")
                        writer.add_scalar(f"charts/episodic_{ts_prefix}_return", infos["episode"]["r"][idx], global_step)
                        writer.add_scalar(f"charts/episodic_{ts_prefix}_length", infos["episode"]["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            student_next_value = student_agent.get_value(next_obs[:args.num_student_envs]).reshape(1, -1)
            teacher_next_ext_value, teacher_next_int_value = teacher_agent.get_value(next_obs[args.num_student_envs:], params = teacher_params)
            teacher_next_ext_value = teacher_next_ext_value.reshape(1, -1)
            teacher_next_int_value = teacher_next_int_value.reshape(1, -1)
            teacher_next_value = teacher_next_ext_value + teacher_next_int_value
            
            next_value = torch.cat([student_next_value, teacher_next_value], axis=1).flatten()
            next_shaped_value = teacher_next_int_value.flatten()
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
                shaped_delta = shaped_rewards[t] + args.gamma * nextshapedvalues * nextnonterminal[args.num_student_envs:] - shaped_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                shaped_advantages[t] = lastshapedgaelam = shaped_delta + args.gamma * args.gae_lambda * nextnonterminal[args.num_student_envs:] * lastshapedgaelam
            returns = advantages + values
            shaped_returns = shaped_advantages + shaped_values

        # Student batch
        b_student_obs = obs[:, :args.num_student_envs].reshape((-1,) + envs.single_observation_space.shape)
        b_student_logprobs = logprobs[:, :args.num_student_envs].reshape(-1)
        b_student_actions = actions[:, :args.num_student_envs].reshape((-1,) + envs.single_action_space.shape)
        b_student_advantages = advantages[:, :args.num_student_envs].reshape(-1)
        b_student_returns = returns[:, :args.num_student_envs].reshape(-1)
        b_student_values = values[:, :args.num_student_envs].reshape(-1)
         
        b_student_inds = np.arange(args.num_student_envs * args.num_steps)
        
        # Teacher update
        b_teacher_obs = obs[:, args.num_student_envs:].reshape((-1,) + envs.single_observation_space.shape)
        b_teacher_logprobs = logprobs[:, args.num_student_envs:].reshape(-1)
        b_teacher_actions = actions[:, args.num_student_envs:].reshape((-1,) + envs.single_action_space.shape)
        b_teacher_advantages = advantages[:, args.num_student_envs:].reshape(-1)
        b_teacher_returns = returns[:, args.num_student_envs:].reshape(-1)
        b_teacher_values = values[:, args.num_student_envs:].reshape(-1)
        
        b_teacher_shaped_advantages = shaped_advantages.reshape(-1)
        b_teacher_shaped_returns = shaped_returns.reshape(-1)
        b_teacher_shaped_values = shaped_values.reshape(-1)
        
        b_teacher_params = params.reshape(-1, args.num_shaping)

        # Optimizing the policy and value network
        b_teacher_inds = np.arange(args.num_teacher_envs * args.num_steps)
        
        b_inds = torch.cat([
            torch.cat([torch.tensor(b_student_inds).unsqueeze(1), torch.zeros(len(b_student_inds), 1)], dim=1),
            torch.cat([torch.tensor(b_teacher_inds).unsqueeze(1), torch.ones(len(b_teacher_inds), 1)], dim=1)
        ], dim=0)

        clipfracs = []
        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            b_inds = b_inds[torch.randperm(b_inds.size(0))]
            for start in range(0, args.num_envs * args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_student_inds = mb_inds[mb_inds[:, 1] == 0][:, 0].long()
                mb_teacher_inds = mb_inds[mb_inds[:, 1] == 1][:, 0].long()
                
                # Student update
                
                if len(mb_student_inds) > 1:
                    
                    _, studentlogprob, studententropy, studentnewvalue = student_agent.get_action_and_value(b_student_obs[mb_student_inds], action = b_student_actions.long()[mb_student_inds])
                    
                    studentratio = (studentlogprob - b_student_logprobs[mb_student_inds]).exp()
                    
                    mb_student_advantages = b_student_advantages[mb_student_inds]
                    if args.norm_adv:
                        mb_student_advantages = (mb_student_advantages - mb_student_advantages.mean()) / (mb_student_advantages.std() + 1e-8)
                        
                    # Policy loss
                    
                    studentpg_loss1 = -studentratio * mb_student_advantages
                    studentpg_loss2 = -torch.clamp(studentratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_student_advantages
                    studentpg_loss = torch.max(studentpg_loss1, studentpg_loss2).mean()
                    
                    # Value loss
                    
                    studentnewvalue = studentnewvalue.reshape(-1)
                    
                    if args.clip_vloss:
                        studentv_loss_unclipped = (studentnewvalue - b_student_returns[mb_student_inds]) ** 2
                        studentv_clipped = b_student_values[mb_student_inds] + torch.clamp(
                            studentnewvalue - b_student_values[mb_student_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        studentv_clipped = studentv_clipped
                        studentv_loss_clipped = (studentv_clipped - b_student_returns[mb_student_inds]) ** 2
                        studentv_loss_max = torch.max(studentv_loss_unclipped, studentv_loss_clipped)
                        studentv_loss = 0.5 * studentv_loss_max.mean()
                    else:
                        studentv_loss = 0.5 * ((studentnewvalue - b_student_returns[mb_student_inds]) ** 2).mean()

                    studententropy_loss = studententropy.mean()
                    loss = studentpg_loss - args.ent_coef * studententropy_loss + studentv_loss * args.vf_coef
                else:
                    loss = 0.0
                    
                if len(mb_teacher_inds) > 1:
                    
                    _, studentlogprob, studententropy, studentnewvalue = student_agent.get_action_and_value(b_teacher_obs[mb_teacher_inds], action = b_teacher_actions.long()[mb_teacher_inds])
                    _, teacherlogprob, teacherentropy, teachernewextvalue, teachernewintvalue = teacher_agent.get_action_and_value(b_teacher_obs[mb_teacher_inds], action = b_teacher_actions.long()[mb_teacher_inds], params = b_teacher_params[mb_teacher_inds])
                    teachernewvalue = teachernewextvalue + teachernewintvalue
                    
                    studentratio = (studentlogprob - b_teacher_logprobs[mb_teacher_inds]).exp()
                    teacherratio = (teacherlogprob - b_teacher_logprobs[mb_teacher_inds]).exp()
                    
                    mb_teacher_advantages = b_teacher_advantages[mb_teacher_inds]
                    mb_teacher_shaped_advantages = b_teacher_shaped_advantages[mb_teacher_inds]
                    
                    if args.norm_adv:
                        mb_teacher_advantages = (mb_teacher_advantages - mb_teacher_advantages.mean()) / (mb_teacher_advantages.std() + 1e-8)
                        mb_teacher_shaped_advantages = (mb_teacher_shaped_advantages - mb_teacher_shaped_advantages.mean()) / (mb_teacher_shaped_advantages.std() + 1e-8)
                        
                    # Policy loss
                    
                    teacherpg_loss1 = -teacherratio * mb_teacher_advantages
                    teacherpg_loss2 = -torch.clamp(teacherratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_teacher_advantages
                    teacherpg_loss = torch.max(teacherpg_loss1, teacherpg_loss2).mean()
                    
                    # Value loss
                    
                    teachernewvalue = teachernewvalue.reshape(-1)
                    teachernewintvalue = teachernewintvalue.reshape(-1)
                    
                    if args.clip_vloss:
                        # External value loss
                        teacherv_loss_unclipped = (teachernewvalue - b_teacher_returns[mb_teacher_inds]) ** 2
                        teacherv_clipped = b_teacher_values[mb_teacher_inds] + torch.clamp(
                            teachernewvalue - b_teacher_values[mb_teacher_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        teacherv_clipped = teacherv_clipped
                        teacherv_loss_clipped = (teacherv_clipped - b_teacher_returns[mb_teacher_inds]) ** 2
                        teacherv_loss_max = torch.max(teacherv_loss_unclipped, teacherv_loss_clipped)
                        teacherv_loss = 0.5 * teacherv_loss_max.mean()
                        
                        # Internal value loss
                        teacherintv_loss_unclipped = (teachernewintvalue - b_teacher_shaped_returns[mb_teacher_inds]) ** 2
                        teacherintv_clipped = b_teacher_shaped_values[mb_teacher_inds] + torch.clamp(
                            teachernewintvalue - b_teacher_shaped_values[mb_teacher_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )   
                        teacherintv_clipped = teacherintv_clipped
                        teacherintv_loss_clipped = (teacherintv_clipped - b_teacher_shaped_returns[mb_teacher_inds]) ** 2
                        teacherintv_loss_max = torch.max(teacherintv_loss_unclipped, teacherintv_loss_clipped)
                        teacherintv_loss = 0.5 * teacherintv_loss_max.mean()
                    else:
                        teacherv_loss = 0.5 * ((teachernewvalue - b_teacher_returns[mb_teacher_inds]) ** 2).mean()
                        teacherintv_loss = 0.5 * ((teachernewintvalue - b_teacher_shaped_returns[mb_teacher_inds]) ** 2).mean()
                        
                    teacherentropy_loss = teacherentropy.mean()
                    loss += teacherpg_loss - args.ent_coef * teacherentropy_loss + (teacherv_loss + teacherintv_loss) * args.vf_coef
                    
                    # Importance sampling loss
                    
                    mb_gen_weight = state_generator.param_ratio(b_teacher_obs[mb_teacher_inds], b_teacher_params[mb_teacher_inds])
                    mb_norm_gen_weight = mb_gen_weight.softmax(dim=1)
                    
                    # RP-DRO
                    norm_gen_loss = -torch.sum(mb_norm_gen_weight * torch.log(teacherratio + 1e-8))
                    
                    # Imitation loss
                    
                    pdl_term1 = mb_teacher_advantages * studentratio
                    pdl_term2 = mb_teacher_advantages * torch.clamp(studentratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    
                    il_loss1 = -mb_gen_weight.clone().detach() * pdl_term1
                    il_loss2 = -mb_gen_weight.clone().detach() * pdl_term2
                    il_loss = torch.max(il_loss1, il_loss2).mean()
                    
                    # Importance sampling loss - maximize gap
                    
                    is_loss1 = -mb_gen_weight * pdl_term1.clone().detach()
                    is_loss2 = -mb_gen_weight * pdl_term2.clone().detach()
                    is_loss = -torch.max(is_loss1, is_loss2).mean()
                    
                    loss += args.il_coef * il_loss + args.is_coef * is_loss + args.kl_imp_coef * norm_gen_loss
            
                student_optimizer.zero_grad()
                teacher_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_agent.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(teacher_agent.parameters(), args.max_grad_norm)
                student_optimizer.step()
                teacher_optimizer.step()

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", teacher_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", teacherv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", teacherpg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if global_step - last_save > args.save_step or global_step >= args.total_timesteps:
            last_save = global_step
            torch.save(student_agent.state_dict(), f"models/{run_name}/student_agent_{iteration}.pth")
            torch.save(teacher_agent.state_dict(), f"models/{run_name}/teacher_agent_{iteration}.pth")

    envs.close()
    writer.close()