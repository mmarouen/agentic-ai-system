import numpy as np
import torch
import torch.optim as optim
from gymnasium import Env 
import collections
from torch.utils.tensorboard import SummaryWriter
from lib.commons.utils import Logger, clear_memory
from lib.commons.normalizer import ObsNormalize
from .buffers import ExperienceBuffer
from .agents import Agent, DQN
from .losses import calc_loss


def dqn(
        logger: Logger, writer: SummaryWriter,
        state_dim: int, actions_dim: int, is_atari: bool,
        solved_threshold: float, is_discrete: bool, device: torch.device,
        frame_skip: int, env: Env,
        buffer_size: int, per_alpha: float,  per: bool, d_dqn: bool,
        gamma: float, n_steps: int, learning_rate: float,
        epsilon_start: float, total_steps: int, epsilon_final: float=0.01, 
        batch_size: int=64, warmup_steps: int=20_000, train_freq: int=1, 
        per_beta_update_freq: int=1000, per_beta_update_frequency: int=1000,
        per_beta_steps: int=100_000, per_beta: float = 0.4, reward_window: int=100,
        epsilon_decay_steps: int=300_000, sync_target_net_steps: int = 10_000, clipping_range: float=10
        ):

    def train(current_step):
        optimizer.zero_grad()
        batch = agent.exp_buffer.sample(batch_size, device=device)
        loss_t, probs, actions = calc_loss(
            batch, net, tgt_net, agent.exp_buffer.beta,
            len(agent.exp_buffer.buffer),
            agent.exp_buffer.sum_priorities,
            d_dqn, per, gamma, n_steps)
        writer.add_scalar("loss", loss_t.item(), current_step)
        loss_t.backward()
        ave_norm_value = logger.log_gradients(net)
        writer.add_scalar("grad norm", ave_norm_value, current_step)
        torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_range)
        optimizer.step()
        q_values_dist = {f'Act_{i}': sum(actions == i) for i in range(actions_dim)}
        writer.add_scalars('Q_vals', q_values_dist, current_step)
        if per:
            agent.exp_buffer.update_priorities(probs)
            if current_step % per_beta_update_frequency == 0:
                agent.exp_buffer.update_beta()
        del batch, loss_t, probs, actions

    net = DQN(state_dim, actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
    logger.log_parameters('Value net', str(net))
    tgt_net = DQN(state_dim, actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
    buffer = ExperienceBuffer(buffer_size, per_alpha, per_beta, per_beta_steps, per_beta_update_freq, per)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    normalizer = ObsNormalize(state_dim)
    if not is_atari:
        normalizer.warmup(env)
    agent = Agent(env, buffer, gamma, n_steps, normalizer, is_atari)

    total_rewards = collections.deque(maxlen=reward_window)
    episode_lengths = collections.deque(maxlen=reward_window)
    annealed_epsilon = False
    buffer_filled = False

    logger.log_parameters('Step 0', 'Warmup started')
    while len(agent.exp_buffer) < warmup_steps:
        agent.play_n_steps(net, epsilon_start, device=device)
    start_training = step = agent.steps

    logger.log_parameters(f'Step {agent.steps:,}', 'Finished warmup')
    epsilon_decay_steps = 300_000 if (is_atari or actions_dim >= 4) else 100_000
    sync_target_net_steps = 10_000 if (is_atari or actions_dim >= 4) else 1000
    clipping_range = 10. if is_atari else 5.
    while step < total_steps:
        net.train()
        log_header = f'Step {agent.steps:,}'
        epsilon = max(epsilon_final, epsilon_start - (step - start_training) / epsilon_decay_steps)
        if epsilon <= epsilon_final and not annealed_epsilon:
            annealed_epsilon = True
            logger.log_parameters(log_header, 'Epsilon annealing finished')

        if len(agent.exp_buffer) >= buffer_size and not buffer_filled:
            buffer_filled = True
            logger.log_parameters(log_header, 'Buffer filled')
        reward, episode_length = agent.play_n_steps(net, epsilon, device=device)
        if reward is not None:
            episode_lengths.append(episode_length)
            total_rewards.append(reward)
            writer.add_scalar("epsilon", epsilon, step)
            writer.add_scalar("reward moing average (100 ganes)", np.mean(total_rewards), step)
            writer.add_scalar("episodic return", reward, step)
            writer.add_scalar("episode length moing average (100 ganes)", np.mean(episode_lengths), step)
            writer.add_scalar("episodic length", episode_length, step)
            logger.n_finished_episodes += 1
            if logger.max_ave_reward < np.mean(total_rewards):
                reward_update = f'Ave reward updated {logger.max_ave_reward:.3f} -> {np.mean(total_rewards):.3f}. Saving model...'
                logger.log_parameters(log_header, reward_update)
                logger.save(policy_model=net, normalizer=normalizer)
                logger.max_ave_reward = np.mean(total_rewards)
                logger.argmax_ave_steps = step
            if logger.max_ave_reward >= solved_threshold:
                close_msg = f'Best score {logger.max_ave_reward} reached within {logger.argmax_abs_steps} steps, {logger.n_finished_episodes} episodes'
                logger.log_parameters(log_header, close_msg)
                break
        summary_dict = {
            "ep reward": reward,
            "ep length": episode_length,
            "ave reward": np.mean(total_rewards),
            "epsilon": epsilon,
            "Global stats": "------------------",
            "max reward": logger.max_ave_reward,
            "argmax reward": logger.argmax_ave_steps,
            "n episodes": logger.n_finished_episodes,
        }
        if is_atari:
            summary_dict["total frames"] = f"{step * frame_skip:,}"
        logger.log_parameters(log_header, summary_dict)

        if step % train_freq == 0:
            train(step)
        if step % sync_target_net_steps == 0:
            tgt_net.load_state_dict(net.state_dict())
        if step % 1000 == 0:
            logger.log_parameters(log_header, clear_memory())
        step = agent.steps
