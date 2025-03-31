import numpy as np
import torch
from .agents import Agent

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
    policy_net: Agent, entropy_coef: float, epsilon: float
):
    if policy_net.is_discrete:
        actions = actions.long()
    _, log_probs, entropies = policy_net.step_policy(states, action=actions)
    ratio = torch.exp(log_probs - old_logp)
    clip_adv = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    # loss_pi = -(torch.min(ratio * advantages, clip_adv) + entropy_coef * entropies).mean()
    return -(torch.min(ratio * advantages, clip_adv) + entropy_coef * entropies).mean()

def calc_value_loss(states: torch.Tensor, reward_togo: torch.Tensor, value_net: Agent):
    return 0.5 * ((value_net.get_value(states) - reward_togo) ** 2).mean()
    # return F.smooth_l1_loss(value_net(states_v), reward_togo_v, reduction='mean')


def calc_vpg_loss(
        advantages: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, policy_net: Agent,
        entropy_coef: float
):
    if policy_net.is_discrete:
        actions = actions.long()
    _, log_probs, entropies = policy_net.step_policy(states, action=actions)
    return -(log_probs * advantages + entropies * entropy_coef).mean()
