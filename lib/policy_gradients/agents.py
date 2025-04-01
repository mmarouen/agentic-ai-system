import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from lib.commons.nnet import CoreNet

class Agent(CoreNet):
    def __init__(self, state_dim, action_dim, is_atari: bool, is_discrete: bool = False, hidden_states: int = 64):
        super().__init__(action_dim=action_dim, state_dim=state_dim, is_atari=is_atari, is_discrete=is_discrete, hidden_states=hidden_states)
        if self.is_discrete:
            self.policy = nn.Linear(self.hidden_state_fc, action_dim)
        else:
            self.mean = nn.Linear(self.hidden_state_fc, action_dim)
            log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.init_linear_layers()

    def forward_policy(self, x: torch.Tensor):
        x = self.forward(x)
        if self.is_discrete:
            x = self.policy(x)
            last_layer = Categorical(logits=x)
        else:
            mu = self.mean(x)
            std = torch.exp(self.log_std)
            last_layer = Normal(mu, std)
        return last_layer

    def step_policy(self, state: torch.Tensor, action:torch.Tensor=None, compute_logprob:bool=True):
        pi_t = self.forward_policy(state)
        log_prob = 0.
        if action is None:
            action = pi_t.sample()
        entropy = pi_t.entropy()
        if self.is_discrete:
            if compute_logprob:
                log_prob = pi_t.log_prob(action)
            action_to_env = action.detach().cpu().numpy()
        else:
            if compute_logprob:
                log_prob = pi_t.log_prob(action).sum(dim=-1)
            action_to_env = action.detach().cpu().numpy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)
        return action_to_env, log_prob, entropy