import gymnasium as gym
import ale_py
import numpy as np
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import os
import collections
import shutil
import torch
import torch.optim as optim
import time
import yaml
from lib.commons.wrappers import make_env
from lib.commons.utils import Logger, clear_memory
from lib.qlearn.agents import DQN, Agent
from lib.qlearn.utils import ExperienceBuffer
from lib.qlearn.losses import calc_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', type=str, default='CartPole-v1')
    parser.add_argument('--video-record_freq', type=int, default=50_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-window', type=int, default=100)
    parser.add_argument('--buffer-size', type=int, default=30_000)
    parser.add_argument('--warmup-steps', type=int, default=20_000)
    parser.add_argument('--epsilon-start', type=float, default=0.1)
    parser.add_argument('--epsilon-final', type=int, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--train-freq', type=int, default=5)
    parser.add_argument('--epsilon-decay-steps', type=int, default=500_000)
    parser.add_argument('--sync-target-net-steps', type=int, default=10_000)
    parser.add_argument('--n-steps', type=int, default=1)
    parser.add_argument('--d-dqn', type=bool, default=False)
    parser.add_argument('--per', type=bool, default=False)
    parser.add_argument('--per-alpha', type=float, default=0.6)
    parser.add_argument('--per-beta', type=float, default=0.4)
    parser.add_argument('--per-beta-steps', type=int, default=100_000)
    parser.add_argument('--per-beta-update-freq', type=int, default=1_000)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--total-steps', type=int, default=10_000_000)
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--exp-desc', type=str, default='')
    args = parser.parse_args()

    timestamp = time.strftime('%y%m%d%H%M')
    map_name = args.map_name.replace(r'/', '-')
    ROOT_NAME = "runs/" + map_name + f'-dqn-{timestamp}/'
    MODEL_FILE = ROOT_NAME + map_name + "-best.dat"
    LOG_FILE = ROOT_NAME + map_name + "-LOG.txt"
    VIDEO_FOLDER = ROOT_NAME + "video_train"
    TENSORBOARD_FOLDER = ROOT_NAME + "tensorboard"

    if os.path.exists(ROOT_NAME):
        shutil.rmtree(ROOT_NAME)
    os.mkdir(ROOT_NAME)


    with open('config/games.yaml') as f:
        GAMES = yaml.safe_load(f)
    SOLVED_REWARD = GAMES[args.map_name]['reward']
    is_atari = GAMES[args.map_name].get('atari', False)
    device = torch.device("cuda")
    writer = SummaryWriter(TENSORBOARD_FOLDER)
    env = make_env(
            env_name=args.map_name, is_atari=is_atari,
            record_video_freq=args.video_record_freq,
            env_id=0, video_root=VIDEO_FOLDER, eval_mode=False,
            frame_skip=GAMES[args.map_name].get('frame_skip', 4),
            tll=GAMES[args.map_name].get('terminal_on_life_loss', True),
            frame_stacking=GAMES[args.map_name].get('frame_stacking', 4))()

    if isinstance(env.action_space, gym.spaces.Box):
        is_discrete = False
        actions_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        actions_dim = env.action_space.n
    state_dim = env.observation_space.shape

    if not is_discrete:
        raise Exception(f"DQN only supports discrete action spaces, {args.map_name} has a continuous action space")

    logger = Logger(MODEL_FILE, log_file=LOG_FILE)
    logger.log_parameters('Input params', args.__dict__)
    net = DQN(state_dim, actions_dim).to(device)
    logger.log_parameters('Value net', str(net))
    model_layers = []
    for name, param in net.named_parameters():
        if 'bias' not in name:
            model_layers.append(name)

    tgt_net = DQN(state_dim, actions_dim).to(device)
    buffer = ExperienceBuffer(args.buffer_size, args.per_alpha, args.per_beta, args.per_beta_steps, args.per_update_freq, args.per)
    agent = Agent(env, buffer, args.gamma, args.n_steps)
    epsilon = args.epsilon_start
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    total_rewards = collections.deque(maxlen=args.reward_window)
    episode_lengths = collections.deque(maxlen=args.reward_window)
    ts_frame = 0
    ts = time.time()
    speed = 0
    probs = None
    annealed_epsilon = False
    buffer_filled = False

    print('Starting warmup phase')
    while len(agent.exp_buffer) < args.warmup_size_iterations:
        agent.play_n_steps(net, epsilon, device=device)
    step = agent.steps
    start_training = step
    # debug_signal = args.debug_signal_frequency_iterations
    logger.log_parameters(f'Step {agent.total_steps}', 'Finished warmup')
    while step < args.total_steps:
        net.train()
        log_header = f'Step {agent.total_steps}'
        epsilon = max(args.epsilon_final, args.epsilon_start - (step - start_training) / args.epsilon_decay_steps)
        current_time = time.strftime('%y-%m-%d:%Hh%Mm')
        if epsilon <= args.epsilon_final and not annealed_epsilon:
            annealed_epsilon = True
            logger.log_parameters(log_header, 'Epsilon annealing finished')

        if len(agent.exp_buffer) >= args.buffer_size and not buffer_filled:
            buffer_filled = True
            logger.log_parameters(log_header, 'Buffer filled')

        reward, episode_length = agent.play_n_steps(net, epsilon, device=device)
        if reward is not None:
            episode_lengths.append(episode_length)
            total_rewards.append(reward)
            writer.add_scalar("epsilon", epsilon, step)
            writer.add_scalar("fps", speed, step)
            writer.add_scalar("reward moing average (100 ganes)", np.mean(total_rewards), step)
            writer.add_scalar("episodic return", reward, step)
            writer.add_scalar("episode length moing average (100 ganes)", np.mean(episode_lengths), step)
            writer.add_scalar("episodic length", episode_length, step)
            logger.n_finished_episodes += 1
            if logger.max_ave_reward < np.mean(total_rewards):
                reward_update = f'Ave reward updated {logger.max_ave_reward:.3f} -> {summary_dict["ave reward"]:.3f}. Saving model...'
                logger.log_parameters(log_header, reward_update)
                torch.save(net.state_dict(), MODEL_FILE)
                logger.max_ave_reward = np.mean(total_rewards)
                logger.argmax_abs_steps = step
            if logger.max_ave_reward >= SOLVED_REWARD:
                close_msg = f'Best score {logger.max_ave_reward} reached within {logger.argmax_abs_steps} steps, {logger.n_finished_episodes} episodes'
                logger.log_parameters(log_header, close_msg)
                break
        summary_dict = {
            "ep reward": reward,
            "ep length": episode_length,
            "mean reward": np.mean(total_rewards),
            "Global stats": "------------------",
            "max reward": logger.max_ave_reward,
            "argmax reward": logger.argmax_ave_steps,
            "n episodes": logger.n_finished_episodes,
            "total frames": step * GAMES[args.map_name].get('frame_skip', 4)
        }
        logger.log_parameters(log_header, summary_dict)

        if step % args.train_freq == 0:
            optimizer.zero_grad()
            batch = agent.exp_buffer.sample(args.batch_size, device=device)
            loss_t, probs = calc_loss(
                batch, net, tgt_net, agent.exp_buffer.beta,
                len(agent.exp_buffer.buffer),
                agent.exp_buffer.sum_priorities,
                args.d_dqn, args.per, args.gamma, args.n_steps)
            writer.add_scalar("loss", loss_t.item(), step)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            if args.per:
                agent.exp_buffer.update_priorities(probs)
                if step % args.per_beta_update_frequency == 0:
                    agent.exp_buffer.update_beta()

        if step % args.sync_target_net_steps == 0:
            tgt_net.load_state_dict(net.state_dict())
        # debug loop
        '''
        if iteration_idx >= debug_signal:
            debug_signal += hyperparams.debug_signal_frequency_iterations
            # logging Q-values distribution
            states_v, _, _, _, _, _ = agent.exp_buffer.sample(100, device=device, debug_mode=True)
            net.eval()
            q_vals = net(states_v).cpu().detach().numpy()
            n_actions = q_vals.shape[1]
            dist = np.argmax(q_vals, axis=1)
            q_values_dist = {f'Act_{i}': sum(dist == i) for i in range(n_actions)}
            writer.add_scalars('Q_vals', q_values_dist, frame_idx)

            # logging gradient abs mean values for a selected random layers
            sample_params_gradients = {f'L_{i}': net.get_parameter(model_layers[i]).grad.abs().mean().item() for i in range(len(model_layers))}
            writer.add_scalars('Grads', sample_params_gradients, frame_idx)
            rouneded_values = {k: round(v, 5) for k, v in sample_params_gradients.items()}
            if best_mean_reward is None:
                print(f"Iteration {iteration_idx}, frame {frame_idx}, {len(total_rewards)} games, \
{current_time}, epsilon {epsilon:.4f}:\nbest mean reward= None, latest average reward: None")
            else:
                print(f"It {iteration_idx}, fr {frame_idx}, {len(total_rewards)} games, \
{current_time}, eps {epsilon:.4f}:\n---max reward so far {agent.max_reward} ({agent.max_reward_freq} times), average mean length {mean_length:.4f},\
\n---best mean reward {best_mean_reward:.4f},\
latest average reward {mean_reward:.4f}, latest average scaled reward {mean_scaled_reward:.4f}")
            del states_v'
        '''
        del batch, loss_t, probs
        if step % 1000 == 0:
            clear_memory()
        step = agent.steps
    writer.close()