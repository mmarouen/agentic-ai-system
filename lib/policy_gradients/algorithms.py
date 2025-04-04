import torch
import os
import random
import collections
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from .core import rollout, RolloutInfo, learn
from .losses import calc_ppo_loss, calc_total_loss, calc_value_loss, calc_vpg_loss
from lib.commons.utils import Logger
from lib.commons.normalizer import ObsNormalize
from .agents import Agent


def policy_gradient(
        main_env: gym.vector.VectorEnv, test_env: gym.Env, logger: Logger, is_atari: bool, solved_threshold: float, frame_skip_count: int,
        device: torch.device, writer: SummaryWriter, seed: int, shared_net: bool,
        algorithm: str='VPG', gamma:float=0.99, lambda_:float=0.97, epochs:int=4000, steps_per_env:float=8000, train_value_iter:int=40,
        train_policy_iter:int=40, lr_policy: float=1e-3, lr_value: float=3*1e-4, epsilon:float=0.2,
        entropy_init: float=1e-3, entropy_final:float = 1e-3, smoothing_reward_window: int=20):

    if isinstance(main_env.single_action_space, gym.spaces.Box):
        is_discrete = False
        actions_dim = main_env.single_action_space.shape[0]
    elif isinstance(main_env.single_action_space, gym.spaces.Discrete):
        is_discrete = True
        actions_dim = main_env.single_action_space.n
    state_dim = main_env.single_observation_space.shape

    num_minibatches = 4
    minibatch_size = int(steps_per_env * main_env.num_envs / num_minibatches)
    ppo = algorithm == 'ppo'
    if not ppo:
        train_policy_iter = 1
        train_value_iter = 1
        shared_net = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    normalizer = ObsNormalize(state_dim)
    if not is_atari:
        normalizer.warmup(test_env)

    if shared_net:
        agent = Agent(state_dim=state_dim, action_dim=actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
        logger.log_parameters('Agent nnet', str(agent))
        optimizer = optim.Adam(agent.parameters(), lr=lr_policy, eps=1e-5)
        policy_net = value_net = optimizer_policy = optimizer_value = None
    else:
        # policy network init
        policy_net = Agent(state_dim, actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
        logger.log_parameters('policy_nnet', str(policy_net))
        optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)
        # value network init
        value_net = Agent(state_dim, actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
        logger.log_parameters('value_nnet', str(value_net))
        optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value, eps=1e-5)
        agent = optimizer = None

    converged = False
    n_steps = steps_per_env * main_env.num_envs
    episode_info = RolloutInfo(
        steps_per_env=steps_per_env,
        is_atari=is_atari,
        ppo=ppo,
        shared_net=shared_net,
        rews=np.zeros((main_env.num_envs,), dtype=np.float32),
        lengths=np.zeros((main_env.num_envs,), dtype=np.int32),
        episode_rewards=collections.deque(maxlen=smoothing_reward_window),
        episode_lengths=collections.deque(maxlen=smoothing_reward_window)
        )

    previous_reward = logger.max_ave_reward
    current_states, _ = main_env.reset(seed=[seed + i for i in range(main_env.num_envs)])
    for epoch in range(epochs):
        current_entropy_coef = 0.
        if entropy_init != 0.:
            current_entropy_coef = entropy_init - epoch * (entropy_init - entropy_final) / epochs
        current_learning_rate = (1.0 - (epoch - 1.0) / epochs) * lr_policy
        if shared_net:
            optimizer.param_groups[0]["lr"] = current_learning_rate
        else:
            optimizer_policy.param_groups[0]["lr"] = current_learning_rate
            optimizer_value.param_groups[0]["lr"] = current_learning_rate
        if not is_atari:
            normalizer.unfreeze()
        obs, act, logp, rewards, dones, vals,\
        truncs, next_obs, last_obs, last_done,\
        episode_info = rollout(main_env, current_states, normalizer,
                               episode_info, writer, device,
                               agent, policy_net, value_net)
        rtg, adv = learn(
            last_obs=last_obs,
            last_done=last_done,
            rewards=rewards,
            dones=dones,
            truncs=truncs,
            next_obs=next_obs,
            vals=vals,
            ep_info=episode_info,
            gamma=gamma,
            lambda_=lambda_,
            agent=agent if shared_net else value_net,
            device=device
            )
        # flatten input
        flat_adv = torch.as_tensor(adv.reshape(-1)).to(device)
        flat_act = torch.as_tensor(act.reshape((-1,) + main_env.single_action_space.shape)).to(device)
        flat_logp = torch.as_tensor(logp.reshape(-1)).to(device)
        flat_vals = torch.as_tensor(vals.reshape(-1)).to(device)
        flat_rtg = torch.as_tensor(rtg.reshape(-1)).to(device)
        flat_obs = torch.as_tensor(obs.reshape((-1,) + main_env.single_observation_space.shape)).to(device)

        adv_mean, adv_std = flat_adv.mean(), flat_adv.std()
        total_frames_processed = episode_info.total_steps_processed * frame_skip_count
        log_header = f'Epoch {epoch}-Steps {episode_info.total_steps_processed:,}-Frames {total_frames_processed:,}'

        # batch statistics
        if len(episode_info.episode_rewards) > 0:
            summary_dict = {
                "ave reward": np.mean(episode_info.episode_rewards),
                "max reward": np.max(episode_info.episode_rewards),
                "min reward": np.min(episode_info.episode_rewards),
                "ave length": np.mean(episode_info.episode_lengths),
                "max length": np.max(episode_info.episode_lengths),
                "min length": np.min(episode_info.episode_lengths),
                "ave adv": adv_mean.item(),
                "std adv": adv_std.item(),
                "n episodes": len(episode_info.episode_rewards),
                "ave entropy": episode_info.rollout_entropy,
                "-----": 'Global stats  -------',
                "max ave reward": logger.max_ave_reward,
                "argmax ave epoch": logger.argmax_ave_epoch,
                "argmax ave steps": logger.argmax_ave_steps,
                "max abs reward": logger.max_abs_reward,
                "argmax abs epoch": logger.argmax_abs_epoch,
                "argmax abs steps": logger.argmax_abs_steps,
                }
        else:
            summary_dict = {
                "ave reward": previous_reward,
                "max reward": previous_reward,
                "min reward": previous_reward,
                "ave length": previous_reward,
                "max length": previous_reward,
                "min length": previous_reward,
                "ave adv": adv_mean.item(),
                "std adv": adv_std.item(),
                "n episodes": len(episode_info.episode_rewards),
                "ave entropy": episode_info.rollout_entropy,
            }
        if is_atari:
            summary_dict["total atari frames"] = total_frames_processed
        writer.add_scalars('Rewards', {"Average": summary_dict["ave reward"], "Abs max": logger.max_abs_reward}, epoch)
        writer.add_scalar("Average entropy", summary_dict["ave entropy"], epoch)
        writer.add_scalar("Advantage mean", summary_dict["ave adv"], epoch)
        #writer.add_scalar("Values mean", summary_dict["ave value"], epoch)
        #writer.add_scalar("Values std", summary_dict["std value"], epoch)

        if epoch == 0:
            logger.max_ave_reward = summary_dict["ave reward"]
            logger.max_abs_reward = summary_dict["ave reward"]

        if summary_dict["ave reward"] > logger.max_ave_reward:
            reward_update = f'Ave reward updated {logger.max_ave_reward:.3f} -> {summary_dict["ave reward"]:.3f}. Saving model...'
            logger.log_parameters(log_header, reward_update)
            if shared_net:
                logger.save(policy_model=agent, normalizer=normalizer, is_shared=shared_net)
            else:
                logger.save(policy_model=policy_net, normalizer=normalizer, is_shared=shared_net)
            logger.max_ave_reward = summary_dict["ave reward"]
            logger.argmax_ave_epoch = epoch
            logger.argmax_ave_steps = episode_info.total_steps_processed

        if summary_dict["max reward"] > logger.max_abs_reward:
            logger.max_abs_reward = summary_dict["max reward"]
            logger.argmax_abs_epoch = epoch
            logger.argmax_abs_steps = episode_info.total_steps_processed

        if summary_dict["ave reward"] >= solved_threshold:
            converged = True
            print(f'Epoch {epoch} solved: {summary_dict["ave reward"]:.4f}')
            break
        logger.log_parameters(log_header, summary_dict)

        ave_policy_loss = 0
        ave_value_loss = 0
        ave_total_loss = 0
        ave_norm = 0
        ave_norm_value = 0
        n_iter = 0
        flat_adv = (flat_adv - adv_mean) / (adv_std + 1e-8)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        for _ in range(train_policy_iter):
            inds = np.random.permutation(n_steps)
            for start in range(0, n_steps, minibatch_size):
                n_iter += 1
                end = start + minibatch_size
                inds_ = inds[start: end]
                adv_ = flat_adv[inds_]
                act_ = flat_act[inds_]
                obs_ = flat_obs[inds_]
                logp_ = flat_logp[inds_]
                vals_ = flat_vals[inds_]
                rtg_ = flat_rtg[inds_]
                if shared_net:
                    optimizer.zero_grad()
                    total_loss, policy_loss, value_loss = calc_total_loss(
                        actions=act_,
                        advantages=adv_,
                        states=obs_,
                        old_logp=logp_,
                        reward_togo=rtg_,
                        agent=agent,
                        entropy_coef=current_entropy_coef,
                        epsilon=epsilon,
                        old_values=vals_)
                    total_loss.backward()
                    ave_norm += logger.log_gradients(agent)
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer.step()
                    ave_policy_loss += policy_loss
                    ave_value_loss += value_loss
                    ave_total_loss += total_loss.item()
                    del obs_, adv_, act_, logp_, vals_, rtg_, total_loss, policy_loss, value_loss
                else:
                    optimizer_policy.zero_grad()
                    if ppo:
                        loss = calc_ppo_loss(
                            advantages=adv_,
                            states=obs_,
                            old_logp=logp_,
                            policy_net=policy_net,
                            entropy_coef=current_entropy_coef,
                            actions=act_,
                            epsilon=epsilon)
                    else:
                        loss = calc_vpg_loss(
                            advantages=adv_,
                            states=obs_,
                            actions=act_,
                            policy_net=policy_net,
                            entropy_coef=current_entropy_coef
                        )
                    loss.backward()
                    ave_norm += logger.log_gradients(policy_net)
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer_policy.step()
                    ave_policy_loss += loss.item()
                    del obs_, adv_, act_, logp_, vals_, rtg_, loss

        ave_policy_loss /= n_iter
        writer.add_scalar("policy loss", ave_policy_loss, epoch)
        if shared_net:
            ave_value_loss /= n_iter
            ave_norm /= n_iter
            ave_total_loss /= n_iter
            writer.add_scalar("value loss", ave_value_loss, epoch)
            writer.add_scalar("total loss", ave_total_loss, epoch)
            writer.add_scalar("total gradient norm", ave_norm, epoch)
        else:
            writer.add_scalar("policy gradient norm", ave_norm, epoch)

        if not shared_net:
            for _ in range(train_value_iter):
                inds = np.random.permutation(n_steps)
                for start in range(0, n_steps, minibatch_size):
                    n_iter += 1
                    end = start + minibatch_size
                    inds_ = inds[start: end]
                    adv_ = flat_adv[inds_]
                    act_ = flat_act[inds_]
                    obs_ = flat_obs[inds_]
                    logp_ = flat_logp[inds_]
                    vals_ = flat_vals[inds_]
                    rtg_ = flat_rtg[inds_]
                    optimizer_value.zero_grad()
                    loss = calc_value_loss(states=obs_, reward_togo=rtg_, value_net=value_net)
                    loss.backward()
                    ave_norm_value += logger.log_gradients(policy_net)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
                    optimizer_value.step()
                    ave_value_loss += loss.item()
                    del obs_, adv_, act_, logp_, vals_, rtg_, loss
            ave_value_loss /= n_iter
            ave_norm_value /= n_iter
            writer.add_scalar("value loss", ave_value_loss, epoch)
            writer.add_scalar("value gradient norm", ave_norm_value, epoch)

    close_msg = f'Best score {logger.max_ave_reward} reached within {logger.argmax_ave_epoch} epochs ({logger.argmax_ave_steps:,} steps)'
    if converged:
        close_msg += f'\nAlgorithm reached target reward {solved_threshold}'
    else:
        close_msg += f'\nAlgorithm didnt reach target reward {solved_threshold}'
    logger.log_parameters('Closing', close_msg)
    return converged, logger.max_ave_reward, logger.argmax_ave_epoch