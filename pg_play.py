import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import yaml
from lib.policy_gradients.agents import Agent
from lib.commons.utils import play_eval_step, describe_env
from lib.commons.wrappers import make_atari_env
import ale_py

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', type=str, default='Pendulum-v1')
    parser.add_argument('--folder', type=str, default='test')
    args = parser.parse_args()

    with open('config/games.yaml') as f:
        GAMES = yaml.safe_load(f)
    is_atari = GAMES[args.map_name].get('atari', False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_name = args.map_name.replace(r'/', '-')
    ROOT_NAME = f"runs/{map_name}-{args.folder}/"
    NORM_FILE = ROOT_NAME + map_name + '-normalizers.yaml'
    MODEL_FILE = ROOT_NAME + map_name + "-best-policy.dat"
    if not is_atari:
        eval_env = gym.make(args.map_name, render_mode="rgb_array")
    else:
        eval_env = make_atari_env(args.map_name, eval_mode=True, render_mode="rgb_array")
    eval_env = RecordVideo(eval_env, video_folder=ROOT_NAME + "video_play", name_prefix=args.folder.lower())
    actions_dim, state_dim, is_discrete = describe_env(eval_env)
    with open(NORM_FILE) as f:
        my_dict = yaml.safe_load(f)
    mean_obs = np.asarray(my_dict['observation_mean'])
    std_obs = np.asarray(my_dict['observation_std'])
    policy = Agent(state_dim=state_dim, action_dim=actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
    policy.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))

    reward, steps = play_eval_step(
        eval_env=eval_env,
        policy=policy,
        obs_mean=mean_obs,
        obs_std=std_obs,
        device=device)
    print(f'Reward {reward}, total steps {steps}')

