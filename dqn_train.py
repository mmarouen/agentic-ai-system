import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import torch
import time
import yaml
from lib.commons.wrappers import make_env
from lib.commons.utils import Logger
from lib.qlearn.algorithms import dqn

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', type=str, default='Acrobot-v1')
    parser.add_argument('--video-record_freq', type=int, default=20_000)
    parser.add_argument('--exp-name', type=str, default='DQN-test')
    parser.add_argument('--exp-desc', type=str, default='')
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--total-steps', type=int, default=10_000_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer-size', type=int, default=100_000) #1M atari 100K for control
    parser.add_argument('--epsilon-start', type=float, default=1)
    parser.add_argument('--epsilon-final', type=int, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=5*1e-4)
    parser.add_argument('--n-steps', type=int, default=1)
    parser.add_argument('--d-dqn', type=bool, default=False)
    parser.add_argument('--per', type=bool, default=False)
    parser.add_argument('--per-alpha', type=float, default=0.6)
    args = parser.parse_args()

    timestamp = time.strftime('%y%m%d%H%M')
    map_name = args.map_name.replace(r'/', '-')
    ROOT_NAME = f'runs/{map_name}-{args.exp_name}/'
    MODEL_FILE = ROOT_NAME + map_name + "-best.dat"
    LOG_FILE = ROOT_NAME + map_name + "-LOG.txt"
    CONFIG_FILE = ROOT_NAME + map_name + "-normalizers.yaml"
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

    logger = Logger(MODEL_FILE, log_file=LOG_FILE, normalizer_path=CONFIG_FILE)
    logger.log_parameters('Input params', args.__dict__)
    dqn(
        logger, writer, state_dim, actions_dim, is_atari,
        SOLVED_REWARD, is_discrete, device,
        GAMES[args.map_name].get('frame_skip', 5), 
        env, args.buffer_size, args.per_alpha, args.per,
        args.d_dqn, args.gamma, args.n_steps, args.learning_rate,
        args.epsilon_start, args.total_steps)
    writer.close()