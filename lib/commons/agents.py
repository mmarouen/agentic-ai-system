import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Normal

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, is_atari: bool, is_discrete: bool = False, hidden_states: int = 64):
        super(Agent, self).__init__()
        self.is_atari = is_atari
        self.is_discrete = is_discrete
        self.action_dim = action_dim
        if is_atari:
            self.conv = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),

                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            for layer in self.conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            conv_out_size = self._get_conv_out(state_dim)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(state_dim[0], hidden_states),
                nn.ReLU(),
                nn.LayerNorm(hidden_states),
                nn.Linear(hidden_states, hidden_states),
                nn.ReLU(),
                nn.LayerNorm(hidden_states),
                nn.Linear(hidden_states, hidden_states),
                nn.ReLU(),
            )
        self.hidden_state_fc = 512 if is_atari else hidden_states
        self.value = nn.Linear(self.hidden_state_fc, 1)
        if self.is_discrete:
            self.policy = nn.Linear(self.hidden_state_fc, action_dim)
        else:
            self.mean = nn.Linear(self.hidden_state_fc, action_dim)
            log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.init_linear_layers()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        return self.fc(x)

    def init_linear_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                gain = np.sqrt(2)
                if layer.out_features == 1:
                    gain = 1.
                elif layer.out_features == self.action_dim:
                    gain = 0.01
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def get_value(self, x: torch.Tensor):
        return self.value(self.forward(x)).squeeze(-1)

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