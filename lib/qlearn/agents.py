import numpy as np
from typing import Optional
import torch
import torch.nn as nn
from lib.commons.nnet import CoreNet
from lib.commons.normalizer import ObsNormalize
from .buffers import Experience, ExperienceBuffer

class DQN(CoreNet):
    def __init__(self, state_dim, action_dim, is_atari: bool, is_discrete: bool = False, hidden_states: int = 64):
        super().__init__(action_dim=action_dim, state_dim=state_dim, is_atari=is_atari, is_discrete=is_discrete, hidden_states=hidden_states)
        self.value = nn.Linear(self.hidden_state_fc, action_dim)
        self.init_linear_layers()

class Agent:
    def __init__(self, env, exp_buffer: ExperienceBuffer, gamma: float, n_steps: int, normalizer: ObsNormalize, is_atari: bool):
        self.env = env
        self.exp_buffer = exp_buffer
        self.steps = 0
        self.state: Optional[np.ndarray] = None
        self.max_reward = 0.
        self.max_reward_freq = 0
        self.n_steps = n_steps
        self.gamma = gamma
        self.normalizer = normalizer
        self.is_atari = is_atari
        if not is_atari:
            self.normalizer.unfreeze()
        self._reset()

    def _reset(self, **kwargs):
        self.state, *_ = self.env.reset(**kwargs)
        self.total_reward = 0.0
        self.episode_length = 0

    @torch.no_grad()
    def play_n_steps(self, net, epsilon, device: str="cpu"):
        done_reward = None
        episode_length = None
        stored_reward = 0
        is_terminated = False
        action_0 = None
        state_0 = self.state.copy()
        for step in range(self.n_steps):
            action, new_state, is_done, is_terminated, reward = self.play_step(net, epsilon, device)
            if step == 0:
                action_0 = action
            stored_reward += np.clip(reward, -1, 1) * (self.gamma ** step)
            self.total_reward += reward
            if is_terminated:
                exp = Experience(
                    state_0,
                    action_0,
                    stored_reward,
                    is_done,
                    new_state)
                self.exp_buffer.append(exp)
                done_reward = self.total_reward
                episode_length = self.episode_length
                self._reset()
                break
            self.state = new_state
        if not is_terminated:
            exp = Experience(state_0, action_0, stored_reward, is_done, new_state)
            self.exp_buffer.append(exp)
        del state_0, new_state
        return done_reward, episode_length

    @torch.no_grad()
    def play_step(self, net: CoreNet, epsilon=0.0, device="cpu"):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device).unsqueeze(0)
            q_vals_v = net.get_value(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
            del state_v

        # do step in the environment
        new_state, reward, is_done, is_truncated, _ = self.env.step(action)
        new_state = self.normalizer.normalize(new_state) if self.is_atari else new_state
        self.normalizer.update(np.expand_dims(new_state, 0))
        if reward > self.max_reward:
            self.max_reward = reward
            self.max_reward_freq = 1
        elif reward == self.max_reward:
            self.max_reward_freq += 1
        self.steps += 1
        self.episode_length += 1
        return action, new_state, is_done, is_truncated or is_done, reward
