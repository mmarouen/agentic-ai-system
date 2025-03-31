import numpy as np
import gymnasium as gym
import torch
import collections
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from .agents import Agent
from lib.commons.normalizer import ObsNormalize


@dataclass
class RolloutInfo:
    steps_per_env: int
    is_atari: bool
    ppo: bool
    shared_net: bool
    rews: np.ndarray
    lengths: np.ndarray
    episode_rewards: collections.deque
    episode_lengths: collections.deque
    rollout_entropy: float = 0
    global_step: int = 0
    total_frames_processed: int = 0
    total_steps_processed: int = 0

def rollout(
        env: gym.vector.VectorEnv,
        current_states: np.ndarray,
        normalizer: ObsNormalize,
        episode_info: RolloutInfo,
        writer: SummaryWriter,
        device: torch.device,
        agent: Agent,
        policy_net: Agent,
        value_net: Agent
        ):
    obs = np.zeros((episode_info.steps_per_env, env.num_envs) + env.single_observation_space.shape, dtype=np.float32)
    act = np.zeros((episode_info.steps_per_env, env.num_envs) + env.single_action_space.shape, dtype=np.float32)
    logp = np.zeros((episode_info.steps_per_env, env.num_envs), dtype=np.float32)
    rewards = np.zeros((episode_info.steps_per_env, env.num_envs), dtype=np.float32)
    dones = np.zeros((episode_info.steps_per_env, env.num_envs))
    vals = np.zeros((episode_info.steps_per_env, env.num_envs), dtype=np.float32)
    truncs = np.zeros((episode_info.steps_per_env, env.num_envs), dtype=np.bool_)
    next_obs = np.zeros((episode_info.steps_per_env, env.num_envs) + env.single_observation_space.shape, dtype=np.float32)
    last_obs = np.zeros(((env.num_envs,) + env.single_observation_space.shape), dtype=np.float32)
    last_done = np.zeros((env.num_envs,))
    avg_entropy = 0.
    n_steps = episode_info.steps_per_env * env.num_envs
    for step in range(episode_info.steps_per_env):
        if not episode_info.is_atari:
            normalizer.update(current_states)
            normalized_state = normalizer.normalize(current_states)
        else:
            normalized_state = current_states
        state_v = torch.as_tensor(normalized_state, dtype=torch.float32).to(device)
        if episode_info.shared_net:
            actions_env, logprobs, entropies_ = agent.step_policy(state_v, compute_logprob=episode_info.ppo)
            values = agent.get_value(state_v)
        else:
            actions_env, logprobs, entropies_ = policy_net.step_policy(state_v, compute_logprob=episode_info.ppo)
            values = value_net.get_value(state_v)
        new_states, rewards_, dones_, truncs_, infos = env.step(actions_env)
        reshaped_rewards_ = np.clip(rewards_, -1, 1)
        # update batch
        obs[step] = normalized_state.copy()
        act[step] = actions_env
        if episode_info.ppo:
            logp[step] = logprobs.detach().cpu().numpy()
        rewards[step] = reshaped_rewards_
        dones[step] = dones_
        truncs[step] = truncs_
        vals[step] = values.detach().cpu().numpy()
        episode_info.rews += rewards_
        episode_info.lengths += 1
        avg_entropy += entropies_.sum().item()
        if step == episode_info.steps_per_env - 1:
            last_obs = new_states.copy() if episode_info.is_atari else normalizer.normalize(new_states)
            last_done = dones_ * 1
        for idx, done in enumerate(np.logical_or(dones_, truncs_)):
            episode_info.global_step += 1
            if done:
                episode_info.episode_rewards.append(episode_info.rews[idx])
                episode_info.episode_lengths.append(episode_info.lengths[idx])
                writer.add_scalar("episodic_return", episode_info.rews[idx], episode_info.global_step)
                writer.add_scalar("episodic_length", episode_info.lengths[idx], episode_info.global_step)
                episode_info.rews[idx] = 0
                episode_info.lengths[idx] = 0
                if truncs_[idx]:
                    next_valid_obs = infos["final_obs"][idx] if episode_info.is_atari else normalizer.normalize(infos["final_obs"][idx])
                    next_obs[step, idx] = next_valid_obs.copy()
                    if step == episode_info.steps_per_env - 1:
                        last_obs[idx] = next_valid_obs.copy()
        current_states = new_states
    last_obs = torch.as_tensor(last_obs, dtype=torch.float32).to(device)
    avg_entropy /= n_steps
    episode_info.rollout_entropy = avg_entropy
    episode_info.total_steps_processed += n_steps
    return obs, act, logp, rewards, dones, vals, truncs, next_obs, last_obs, last_done, episode_info

def learn(
        last_obs: np.ndarray,
        last_done: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        truncs: np.ndarray,
        next_obs: np.ndarray,
        vals: np.ndarray,
        ep_info: RolloutInfo,
        gamma: float,
        lambda_: float,
        agent: Agent,
        device: torch.device
        ):
    with torch.no_grad():
        next_value = agent.get_value(last_obs)
        adv = np.zeros_like(rewards)
        rtg = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(ep_info.steps_per_env)):
            if t == ep_info.steps_per_env - 1:
                next_non_terminal = 1.0 - last_done
                next_values = next_value.detach().cpu().numpy()
                # rtg[t] = rewards[t] + gamma * next_non_terminal * next_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val_trunc = 0.
                if any(truncs[t + 1]):
                    next_obs_ = torch.as_tensor(next_obs[t + 1]).to(device)
                    next_val_trunc = agent.get_value(next_obs_).detach().cpu().numpy()
                next_values = vals[t + 1] * (1. - truncs[t + 1]) + next_val_trunc * truncs[t + 1]
                # rtg[t] = rewards[t] + gamma * next_non_terminal * rtg[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - vals[t]
            adv[t] = lastgaelam = delta + gamma * lambda_ * next_non_terminal * lastgaelam
        rtg = adv + vals
    return rtg, adv
