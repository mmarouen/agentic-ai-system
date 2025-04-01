import numpy as np
import torch
import gymnasium as gym
from typing import Union
import psutil
import gc
import os
from scipy.signal import lfilter
from .normalizer import ObsNormalize

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

def clear_memory(threshold_gb=6.0, cpu_memory_threshold_mb=20_000):
    """Clears GPU cache if reserved memory exceeds the given threshold."""
    reserved_gb = torch.cuda.memory_reserved() / 1e9  # Convert bytes to GB
    allocated_gb = torch.cuda.memory_allocated() / 1e9  # Bytes to GB
    
    if reserved_gb > threshold_gb:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU Memory cleared! Reserved: {reserved_gb:.2f} GB, Allocated: {allocated_gb:.2f} GB")
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)
    # print(f'current memory usage {mem_usage_mb} MB')
    if mem_usage_mb > cpu_memory_threshold_mb:
        gc.collect()
        print(f"CPU memory usage {mem_usage_mb:.2f} MB exceeds {cpu_memory_threshold_mb} MB, triggering gc.collect()")

class Logger:
    def __init__(self, policy_model_path: str, log_file: str, value_net_path: str=None, normalizer_path: str=None):
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
        self.n_finished_episodes = 0
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
        if normalizer:
            obs_mean, obs_std = normalizer.get_stats()
            config_dict = {
                'observation_mean': obs_mean.tolist(),
                'observation_std': obs_std.tolist(),
                'is_shared': is_shared
            }
            import yaml
            with open(self.normalizer_path, 'w') as outfile:
                yaml.dump(config_dict, outfile, default_flow_style=False)
