import gymnasium as gym
import ale_py
import yaml
from gymnasium.wrappers import RecordVideo
import numpy as np
import numpy as np
import typing as tt
import torch
from lib.commons.wrappers import make_atari_env
from lib.commons.utils import describe_env
from lib.qlearn.agents import DQN


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', type=str, default='Acrobot-v1')
    parser.add_argument('--folder', type=str, default='DQN-test')
    args = parser.parse_args()

    with open('config/games.yaml') as f:
        GAMES = yaml.safe_load(f)
    is_atari = GAMES[args.map_name].get('atari', False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_name = args.map_name.replace(r'/', '-')
    ROOT_NAME = f"runs/{map_name}-{args.folder}/"
    NORM_FILE = ROOT_NAME + map_name + '-normalizers.yaml'
    MODEL_FILE = ROOT_NAME + map_name + "-best.dat"
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

    weights = torch.load(MODEL_FILE, map_location=lambda stg, _: stg, weights_only=True)
    net = DQN(state_dim=state_dim, action_dim=actions_dim, is_atari=is_atari, is_discrete=is_discrete).to(device)
    net.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))

    state, _ = eval_env.reset()
    total_reward = 0.0
    env_steps = 0
    while True:
        env_steps += 1
        state = (state - mean_obs)  / std_obs
        state_v = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32).to(device)
        q_vals = net.get_value(state_v).detach().cpu().numpy()
        action = int(np.argmax(q_vals))
        state, reward, is_done, is_trunc, _ = eval_env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print(f"Total reward: {total_reward:.3f}, {env_steps} environment steps")
    eval_env.close()