import gymnasium as gym
import os
import yaml
import shutil
import torch
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from lib.algorithms import policy_gradient
from lib.wrappers import make_env
from lib.utils import Logger
import ale_py


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--steps_per_env', type=int, default=512) #128 for atari
    parser.add_argument('--video_record_freq', type=int, default=50_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--num_envs', type=int, default=10)
    parser.add_argument('--shared_network', type=bool, default=True)
    parser.add_argument('--steps_total', type=int, default=10_000_000)
    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--exp_name', type=str, default='PPO')
    parser.add_argument('--exp_desc', type=str, default='same as 16, samestep env + decay entropy')
    parser.add_argument('--train_value_iter', type=int, default=4) #5
    parser.add_argument('--train_policy_iter', type=int, default=4) #5
    parser.add_argument('--lr_policy', type=float, default=1e-3) # 2.5*1e-4 for atari
    parser.add_argument('--lr_value', type=float, default=2*1e-3) # 1e-5
    parser.add_argument('--entropy_init', type=float, default=1e-2)
    parser.add_argument('--entropy_final', type=float, default=1e-3)

    args = parser.parse_args()

    with open('config/games.yaml') as f:
        GAMES = yaml.safe_load(f)
    SOLVED_REWARD = GAMES[args.map_name]['reward']
    args.steps_per_epoch = args.steps_per_env * args.num_envs
    args.total_epochs = args.steps_total // args.steps_per_epoch + 1
    is_atari = GAMES[args.map_name].get('atari', False)
    args.num_envs = min(multiprocessing.cpu_count() * 2, args.num_envs)
    map_name = args.map_name.replace(r'/', '-')
    ROOT_NAME = f'runs/{map_name}-{args.exp_name}/'
    POLICY_MODEL_FILE = ROOT_NAME + map_name + "-best-policy.dat"
    VALUE_MODEL_FILE = ROOT_NAME + map_name + "-best-value.dat"
    CONFIG_FILE = ROOT_NAME + map_name + "-normalizers.yaml"
    LOG_FILE = ROOT_NAME + map_name + "-LOG.txt"
    VIDEO_FOLDER = ROOT_NAME + "video_train"
    TENSORBOARD_FOLDER = ROOT_NAME + "tensorboard"

    if os.path.exists(ROOT_NAME):
        shutil.rmtree(ROOT_NAME)
    os.mkdir(ROOT_NAME)
    # if args.video_record_freq > 0: os.mkdir(VIDEO_FOLDER)
    logger = Logger(POLICY_MODEL_FILE, VALUE_MODEL_FILE, normalizer_path=CONFIG_FILE, log_file=LOG_FILE)
    logger.log_parameters('Input params', args.__dict__)
    writer = SummaryWriter(TENSORBOARD_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.vector.SyncVectorEnv(
        [make_env(
            env_name=args.map_name,
            is_atari=is_atari,
            record_video_freq=args.video_record_freq,
            env_id=i,
            video_root=VIDEO_FOLDER,
            eval_mode=False,
            frame_skip=GAMES[args.map_name].get('frame_skip', 4),
            tll=GAMES[args.map_name].get('terminal_on_life_loss', True),
            frame_stacking=GAMES[args.map_name].get('frame_stacking', 4)
            ) for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
    if not is_atari:
        eval_env = gym.make(args.map_name, render_mode="rgb_array")
    else:
        eval_env = make_env(
                args.map_name,
                frame_skip=GAMES[args.map_name].get('frame_skip', 4),
                tll=GAMES[args.map_name].get('terminal_on_life_loss', True),
                frame_stacking=GAMES[args.map_name].get('frame_stacking', 4),
                render_mode='rgb_array'
            )
        f_obs, _ = eval_env.reset()

    policy_gradient(env, eval_env, logger=logger, is_atari=is_atari, device=device, writer=writer, solved_threshold=SOLVED_REWARD,
                    algorithm=args.algorithm, frame_skip_count=GAMES[args.map_name].get('frame_skip', 4), seed=args.seed, shared_net=args.shared_network,
                    gamma=args.gamma, lambda_=args.lambda_, epochs=args.total_epochs, steps_per_env=args.steps_per_env, train_value_iter=args.train_value_iter, 
                    train_policy_iter=args.train_policy_iter, lr_policy=args.lr_policy, lr_value=args.lr_value, epsilon= args.epsilon,
                    entropy_init=args.entropy_init, entropy_final=args.entropy_final)
    writer.close()
    env.close()
