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
        hidden_state_fc = 512 if is_atari else hidden_states
        self.value = nn.Linear(hidden_state_fc, 1)
        if self.is_discrete:
            self.policy = nn.Linear(hidden_state_fc, action_dim)
        else:
            self.mean = nn.Linear(hidden_state_fc, action_dim)
            log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                gain = np.sqrt(2)
                if layer.out_features == 1:
                    gain = 1.
                elif layer.out_features == action_dim:
                    gain = 0.01
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def get_value(self, x: torch.Tensor):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.value(x).squeeze(-1)

    def forward_policy(self, x: torch.Tensor):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        x = self.fc(x)
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

class Value(nn.Module):
    def __init__(self, state_dim, is_atari: bool, hidden_states: int = 64):
        super(Value, self).__init__()
        self.is_atari = is_atari
        if is_atari:
            self.conv = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),

                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )

            conv_out_size = self._get_conv_out(state_dim)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            for layer in self.conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
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
                nn.LayerNorm(hidden_states),
                nn.Linear(hidden_states, 1)
            )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                gain = np.sqrt(2) if layer.out_features != 1 else 1  # Special case for the final layer
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def get_value(self, x):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, is_atari: bool, is_discrete: bool = False, hidden_states: int = 64):
        super(Policy, self).__init__()
        self.is_atari = is_atari
        self.is_discrete = is_discrete
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
                nn.ReLU(),
                # nn.LayerNorm(512),
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
        if self.is_atari:
            hidden_state_fc = 512
        else:
            hidden_state_fc = hidden_states
        if self.is_discrete:
            self.fc = nn.Sequential(*self.fc, nn.Linear(hidden_state_fc, action_dim))
        else:
            self.mean = nn.Linear(hidden_state_fc, action_dim)
            log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                gain = np.sqrt(2) if layer.out_features != action_dim else 0.01  # Special case for the final layer
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)


    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        last_layer = None
        if self.is_discrete:
            last_layer = Categorical(logits=x)
        else:
            mu = self.mean(x)
            # log_std = torch.clamp(self.log_std(x), min=-7, max=1.5)
            std = torch.exp(self.log_std)
            last_layer = Normal(mu, std)
        return last_layer

    def step_policy(self, state_tensor: torch.Tensor, action: torch.Tensor = None, compute_logprob: bool=True):
        pi_t = self.forward(state_tensor)
        log_prob = 0.
        if action is None:
            action = pi_t.sample()
        entropy = pi_t.entropy()
        if self.is_discrete:
            if compute_logprob:
                log_prob = pi_t.log_prob(action)
            action_to_env = action.detach().cpu().numpy()
        else:
            # no squash
            if compute_logprob:
                log_prob = pi_t.log_prob(action).sum(dim=-1)
            action_to_env = action.detach().cpu().numpy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)
        return action_to_env, log_prob, entropy

def calc_total_loss(
        advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, old_logp: torch.Tensor,
        agent: Agent, entropy_coef: float,
        reward_togo: torch.Tensor, old_values: torch.Tensor,
        epsilon: float):

    # policy loss
    if agent.is_discrete:
        actions = actions.long()
    _, log_probs, entropies = agent.step_policy(states, action=actions)
    ratio = torch.exp(log_probs - old_logp)
    clip_adv = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

    # value loss
    new_values = agent.get_value(states)
    value_loss_unclipped =((new_values - reward_togo) ** 2).mean()

    v_clipped = old_values + torch.clamp(new_values - old_values, -epsilon, epsilon)
    value_loss_clipped = (v_clipped - reward_togo) ** 2
    value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
    v_loss = 0.5 * value_loss_max.mean()
    return policy_loss + 0.5 * v_loss - entropy_coef * entropies.mean(), v_loss.item(), policy_loss.item()


def calc_ppo_loss(
    advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, old_logp: np.ndarray,
    policy_net: Policy, entropy_coef: float, epsilon: float
):
    if policy_net.is_discrete:
        actions = actions.long()
    _, log_probs, entropies = policy_net.step_policy(states, action=actions)
    ratio = torch.exp(log_probs - old_logp)
    clip_adv = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    # loss_pi = -(torch.min(ratio * advantages, clip_adv) + entropy_coef * entropies).mean()
    return -(torch.min(ratio * advantages, clip_adv) + entropy_coef * entropies).mean()

def calc_value_loss(states: torch.Tensor, reward_togo: torch.Tensor, value_net: Value):
    return 0.5 * ((value_net.get_value(states) - reward_togo) ** 2).mean()
    # return F.smooth_l1_loss(value_net(states_v), reward_togo_v, reduction='mean')


def calc_vpg_loss(
        advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, policy_net: Policy,
        entropy_coef: float
):
    if policy_net.is_discrete:
        actions = actions.long()
    _, log_probs, entropies = policy_net.step_policy(states, action=actions)
    return -(log_probs * advantages + entropies * entropy_coef).mean()

class ObsNormalize():
    def __init__(self, state_dim: int):
        self.mean = np.zeros(state_dim)
        self.sq_sum = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.min = np.zeros(state_dim)
        self.max = np.zeros(state_dim)
        self.n_obs = 0
        self.epsilon = 1e-8
        self.is_init = True
        self.mean_ = np.zeros(state_dim)
        self.var_ = np.ones(state_dim)
        self.std_ = np.ones(state_dim)

    def unfreeze(self):
        self.mean_ = self.mean
        self.var_ = self.var
        self.std_ = np.sqrt(self.var_)

    def _rolling_update(self, batch_sum, batch_sq, batch_min, batch_max, p):
        if self.is_init:
            self.min = batch_min
            self.max = batch_max
        else:
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)
        total_obs = self.n_obs + p
        batch_mean = batch_sum / p
        batch_var = batch_sq / p - batch_mean ** 2
        delta = batch_mean - self.mean
        self.sq_sum = self.sq_sum + batch_var * p + self.n_obs * p * delta ** 2 / total_obs
        self.mean = (self.n_obs * self.mean + batch_mean * p) / total_obs
        self.n_obs = total_obs
        self.var = self.sq_sum / (self.n_obs - 1)
        self.is_init = False

    def update(self, batch):
        batch_sum = np.sum(batch, axis=0)
        batch_sq = np.sum(batch ** 2, axis=0)
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        size = len(batch)
        self._rolling_update(batch_sum, batch_sq, batch_min, batch_max, size)

    def get_stats(self):
        return self.mean_, np.sqrt(self.var_ + self.epsilon)

    def normalize(self, observation):
        return (observation - self.mean_) / (self.std_ + self.epsilon)