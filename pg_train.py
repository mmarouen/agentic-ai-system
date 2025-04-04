import gymnasium as gym
import os
import yaml
import shutil
import torch
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from lib.policy_gradients.algorithms import policy_gradient
from lib.commons.utils import Logger
from lib.commons.wrappers import make_env
import ale_py


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', type=str, default='Acrobot-v1')
    parser.add_argument('--steps-per-env', type=int, default=128) #128 for atari
    parser.add_argument('--video-record-freq', type=int, default=50_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=10)
    parser.add_argument('--shared-network', type=bool, default=True)
    parser.add_argument('--total-steps', type=int, default=10_000_000)
    parser.add_argument('--algorithm', type=str, default='ppo')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--exp-desc', type=str, default='')
    parser.add_argument('--train-value-iter', type=int, default=4) #5
    parser.add_argument('--train-policy-iter', type=int, default=4) #5
    parser.add_argument('--lr-policy', type=float, default=2.5*1e-4) # 2.5*1e-4 for atari
    parser.add_argument('--lr-value', type=float, default=2*1e-3) # 1e-5
    parser.add_argument('--entropy-init', type=float, default=1e-2)
    parser.add_argument('--entropy-final', type=float, default=1e-3)

    args = parser.parse_args()

    with open('config/games.yaml') as f:
        GAMES = yaml.safe_load(f)
    SOLVED_REWARD = GAMES[args.map_name]['reward']
    args.steps_per_epoch = args.steps_per_env * args.num_envs
    args.total_epochs = args.total_steps // args.steps_per_epoch + 1
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
    logger = Logger(policy_model_path=POLICY_MODEL_FILE, value_net_path=VALUE_MODEL_FILE, normalizer_path=CONFIG_FILE, log_file=LOG_FILE)
    logger.log_parameters('Input params', args.__dict__)
    writer = SummaryWriter(TENSORBOARD_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.vector.AsyncVectorEnv(
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
        eval_env = None
    policy_gradient(env, eval_env, logger=logger, is_atari=is_atari, device=device, writer=writer, solved_threshold=SOLVED_REWARD,
                    algorithm=args.algorithm, frame_skip_count=GAMES[args.map_name].get('frame_skip', 4), seed=args.seed, shared_net=args.shared_network,
                    gamma=args.gamma, lambda_=args.lambda_, epochs=args.total_epochs, steps_per_env=args.steps_per_env, train_value_iter=args.train_value_iter, 
                    train_policy_iter=args.train_policy_iter, lr_policy=args.lr_policy, lr_value=args.lr_value, epsilon= args.epsilon,
                    entropy_init=args.entropy_init, entropy_final=args.entropy_final)
    writer.close()
    env.close()
