import numpy as np
import torch
import gymnasium as gym
from typing import Union
from scipy.signal import lfilter
from dataclasses import dataclass, fields
from .core import ObsNormalize

def describe_env(env: gym.Env):
    actions_dim = None
    if isinstance(env.action_space, gym.spaces.Box):
        is_discrete = False
        actions_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        actions_dim = env.action_space.n
    state_dim = env.observation_space.shape
    return actions_dim, state_dim, is_discrete

def play_eval_step(eval_env: gym.Env, policy: torch.nn.Module, obs_mean: np.ndarray, obs_std: np.ndarray, device: torch.device):
    total_rewards = 0.
    total_steps = 0
    test_obs, _ = eval_env.reset()
    test_done = False
    while not test_done:
        # test_obs = normalizer.normalize(test_obs)
        if not policy.is_atari:
            test_obs = (test_obs - obs_mean) / obs_std
            obs_tensor = torch.as_tensor(test_obs, dtype=torch.float32).to(device)
        else:
            obs_tensor = torch.as_tensor(test_obs, dtype=torch.float32).unsqueeze(0).to(device)
        test_action, _, _ = policy.step_policy(obs_tensor, compute_logprob=False)
        test_action = test_action[0] if policy.is_atari else test_action
        test_obs, r, test_done, test_truncated, _ = eval_env.step(test_action)
        test_done = test_done or test_truncated
        total_rewards += r
        total_steps += 1
    eval_env.close()
    return total_rewards, total_steps

def reshape_reward(map_name: str, reward: Union[float, np.ndarray]):
    if isinstance(reward, np.ndarray):
        reward_ = reward.copy()
    else:
        reward_ = reward
    if map_name == "MountainCar-v0":
        pass
    if map_name == "LunarLander-v3":
        pass
    if map_name == "Pendulum-v1":
        pass
    if "Pong" in map_name:
        pass
    if "BeamRider" in map_name:
        #reward_ = np.clip(reward_, -1., 1.)
        reward_ /= 100
    return reward_

def discount_cumsum_lfilter(rewards, gamma: float):
    return lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]

@dataclass
class Experience:
    action: np.ndarray
    value: float
    reward: float
    normal_reward: float
    state: np.ndarray
    log_prob: float = 0.
    reward_to_go: float = None
    advantage: float = None
    entropy: float = None

    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))

@dataclass
class Episode:
    action_dims: int
    state_dims: tuple
    is_discrete: bool
    sum_entropies: float = 0.
    sum_rewards: float = 0.
    sum_shaped_rewards: float = 0.
    n_steps: int = 0
    sum_adv: float = 0
    sum_adv2: float = 0
    sum_values: float = 0
    sum_values2: float = 0

    def update(self, experience: Experience):
        if self.n_steps == 0:
            if self.is_discrete:
                self.actions = np.empty(10_000, dtype=np.int32)  # 1D array for discrete
                self.sum_actions = np.zeros(1, dtype=np.float32)
                self.sum_actions2 = np.zeros(1, dtype=np.float32)
            else:
                self.actions = np.empty((10_000, self.action_dims), dtype=np.float32)
                self.sum_actions = np.zeros(self.action_dims, dtype=np.float32)
                self.sum_actions2 = np.zeros(self.action_dims, dtype=np.float32)
            self.states = np.empty((10_000,) + self.state_dims, dtype=np.float32)
            self.log_probs = np.empty((10_000,), dtype=np.float32)
            self.rewards = np.empty((10_000,), dtype=np.float32)
            self.values = np.empty((10_000,), dtype=np.float32)
            self.advantages = np.empty((10_000,), dtype=np.float32)

        self.rewards[self.n_steps] = experience.reward
        self.values[self.n_steps] = experience.value
        self.actions[self.n_steps] = experience.action
        self.states[self.n_steps] = experience.state
        self.log_probs[self.n_steps] = experience.log_prob
        self.sum_entropies += experience.entropy
        self.sum_rewards += experience.normal_reward
        self.sum_shaped_rewards += experience.reward
        self.sum_values += experience.value
        self.sum_values2 += experience.value ** 2
        self.sum_actions += experience.action
        self.sum_actions2 += experience.action ** 2
        self.n_steps += 1

    def close(self, gamma:float, lam: float, last_value: float):
        rewards = np.append(self.rewards[:self.n_steps], last_value)
        values = np.append(self.values[:self.n_steps], last_value)
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        reward_togo = discount_cumsum_lfilter(rewards, gamma)[:-1]
        self.advantages = discount_cumsum_lfilter(deltas, gamma * lam)
        self.sum_adv2 = sum(self.advantages ** 2)
        self.sum_adv = sum(self.advantages)
        return self.advantages, reward_togo, self.actions[:self.n_steps], self.states[:self.n_steps], self.log_probs[:self.n_steps], self.values[:self.n_steps]
    
@dataclass
class Batch:
    action_dim: int
    state_dim: tuple
    is_discrete: bool
    average_entropy: float = 0
    average_reward: float = 0
    average_shaped_reward: float = 0
    m_advantage: float = 0
    sum_advantage2: float = 0
    n_steps: int = 0
    sum_values: float = 0
    sum_values2: float = 0
    gamma: float = 0
    lam: float = 0
    n_episodes: int = 0
    max_reward = -1e6
    min_reward = 1e6

    def _update_adv_stats(self, mini_sum, mini_s2, mini_n):
        n_new = self.n_steps + mini_n
        mini_mean = mini_sum / mini_n
        mini_var = mini_s2 / mini_n - (mini_sum / mini_n) ** 2
        delta = mini_mean - self.m_advantage
        self.sum_advantage2 += mini_var * mini_n + self.n_steps * mini_n * delta ** 2 / n_new
        self.m_advantage = (self.n_steps * self.m_advantage + mini_n * mini_mean) / n_new

    def update(self, episode: Episode, last_val: float = 0.):
        if self.n_steps == 0:
            if self.is_discrete:
                self.actions = np.empty(100_000, dtype=np.int32)  # 1D array for discrete
                self.sum_actions = np.zeros(1, dtype=np.float32)
                self.sum_actions2 = np.zeros(1, dtype=np.float32)
            else:
                self.actions = np.empty((100_000, self.action_dim), dtype=np.float32)
                self.sum_actions = np.zeros(self.action_dim, dtype=np.float32)
                self.sum_actions2 = np.zeros(self.action_dim, dtype=np.float32)

            self.states = np.empty((100_000,) + self.state_dim, dtype=np.float32)
            self.advantages = np.empty((100_000, ), dtype=np.float32)
            self.reward_tg = np.empty((100_000, ), dtype=np.float32)
            self.log_probs = np.empty((100_000, ), dtype=np.float32)
            self.rewards = np.empty((100_000,), dtype=np.float32)
            self.values = np.empty((100_000, ), dtype=np.float32)

        adv, rtg, act, stat, logp, vals = episode.close(self.gamma, self.lam, last_value=last_val)
        self.advantages[self.n_steps: self.n_steps + episode.n_steps] = adv
        self.reward_tg[self.n_steps: self.n_steps + episode.n_steps] = rtg
        self.states[self.n_steps: self.n_steps + episode.n_steps] = stat
        self.actions[self.n_steps: self.n_steps + episode.n_steps] = act
        self.log_probs[self.n_steps: self.n_steps + episode.n_steps] = logp
        self.values[self.n_steps: self.n_steps + episode.n_steps] = vals

        # self.episodes.append(episode)
        self._update_adv_stats(episode.sum_adv, episode.sum_adv2, episode.n_steps)
        self.average_shaped_reward += episode.sum_shaped_rewards
        self.average_entropy += episode.sum_entropies / episode.n_steps
        self.average_reward += episode.sum_rewards
        self.n_steps += episode.n_steps
        self.sum_values += episode.sum_values
        self.sum_values2 += episode.sum_values2
        self.sum_actions += episode.sum_actions
        self.sum_actions2 += episode.sum_actions2
        self.max_reward = max(episode.sum_rewards, self.max_reward)
        self.min_reward = min(episode.sum_rewards, self.min_reward)
        self.rewards[self.n_episodes] = episode.sum_rewards
        self.n_episodes += 1

    def get_normalization_stats(self):
        return self.m_advantage, np.sqrt(self.sum_advantage2 / (self.n_steps - 1))

    def get_summary(self):
        d = {}
        d["batch size"] = self.n_steps
        d["episodes per epoch"] = self.n_episodes
        d["ave reward"] = self.average_reward / self.n_episodes
        d["max reward"] = self.max_reward
        d["min reward"] = self.min_reward
        d["ave reward (shaped)"] = self.average_shaped_reward / self.n_episodes
        d["ave entropy"] = self.average_entropy / self.n_episodes
        d["ave value"] = self.sum_values / self.n_steps
        d["std value"] = np.sqrt(self.sum_values2 / self.n_steps - d["ave value"] ** 2)
        d["ave action"] = self.sum_actions / self.n_steps
        d["std action"] = np.sqrt(self.sum_actions2 / self.n_steps - d["ave action"] ** 2)
        return d

    def get(self):
        # return self.advantages[:self.n_steps], self.reward_tg[:self.n_steps], self.actions[:self.n_steps], self.states[:self.n_steps], self.log_probs[:self.n_steps], self.values[:self.n_steps]
        return self.advantages[:self.n_steps], self.reward_tg[:self.n_steps], self.actions[:self.n_steps], self.states[:self.n_steps], self.log_probs[:self.n_steps], self.values[:self.n_steps]

class Logger:
    def __init__(self, policy_model_path: str, value_net_path: str, normalizer_path: str, log_file: str):
        self.policy_model_path = policy_model_path
        self.value_net_path = value_net_path
        self.normalizer_path = normalizer_path
        self.logger_file = log_file
        self.max_ave_reward = -1000
        self.argmax_ave_epoch = 0
        self.argmax_ave_steps = 0
        self.max_abs_reward = -1000
        self.argmax_abs_epoch = 0
        self.argmax_abs_steps = 0
        open(log_file, 'a').close()

    def log_gradients(self, model: torch.nn.Module):
        total_norm = 0.0
        n_params = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                n_params += 1
        total_norm = total_norm ** 0.5
        return total_norm / n_params

    def log_parameters(self, title: str, params):
        ln_title = 48
        if len(title) % 2 == 0:
            title += ' '
        cnt = (ln_title - len(title)) // 2
        display = '-' * cnt + title + '-' * cnt + '|'
        if isinstance(params, dict):
            for k, v in params.items():
                display_ = ''
                if isinstance(v, (int, np.integer)):
                    display_ += f'\n--{k}: {v:,}'
                elif isinstance(v, (np.floating, float)):
                    display_ += f'\n--{k}: {v:,.4f}'
                else:
                    display_ += f'\n--{k}: {str(v)}'
                display_ = display_ + ' ' * (ln_title - len(display_)) + '|'
                display += display_
        else:
            display += f'\n{params}'
        display += '\n' + (ln_title - 1) * '_' + '|'
        export_params = {}
        export_params[title] = params
        with open(self.logger_file, "a") as f:
            f.write(display)
            f.write('\n\n')
        f.close()
        print(display)
        return display

    def save(self, policy_model: torch.nn.Module, value_model: torch.nn.Module=None, normalizer: ObsNormalize=None, is_shared:bool=False):
        torch.save(policy_model.state_dict(), self.policy_model_path)
        if value_model:
            torch.save(value_model.state_dict(), self.value_net_path)
        obs_mean, obs_std = normalizer.get_stats()
        config_dict = {
            'observation_mean': obs_mean.tolist(),
            'observation_std': obs_std.tolist(),
            'is_shared': is_shared
        }
        import yaml
        with open(self.normalizer_path, 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)
