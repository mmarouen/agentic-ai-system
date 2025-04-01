import torch
import numpy as np
import torch.nn as nn

def calc_loss(batch, net, tgt_net, beta, buffer_size, sum_priorities, d_dqn, per, gamma, n_steps):
    states_v, actions_v, rewards_v, done_mask, next_states_v, priorities_v = batch

    state_action_values = net.get_value(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = None
        if d_dqn:
            next_state_acts = net.get_value(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_values = tgt_net.get_value(next_states_v).gather(1, next_state_acts).squeeze(-1)
        else:
            next_state_values = tgt_net.get_value(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * (gamma ** n_steps) + rewards_v
    loss_scalar = None
    losses = 0
    if per:
        probs_v = (priorities_v * buffer_size / sum_priorities) ** (-beta)
        probs_v = probs_v / probs_v.max()
        losses_vector = torch.abs(state_action_values - expected_state_action_values) ** 2
        # losses_vector = torch.abs(state_action_values - expected_state_action_values)
        loss_scalar = torch.sum(losses_vector * probs_v)
        #losses = losses_vector.detach().to('cpu').numpy()
        losses_vector = losses_vector.detach().to('cpu').numpy()
        losses = np.sqrt(losses_vector)

    else:
        loss_scalar = nn.MSELoss()(state_action_values, expected_state_action_values)
        # loss_scalar = nn.L1Loss()(state_action_values, expected_state_action_values)

    return loss_scalar, losses, actions_v.detach().cpu().numpy()
