import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import numpy as np
import numpy as np
import typing as tt
import torch
import collections
from lib import wrappers
from lib import dqn_model

games_setup = {
    "PongNoFrameskip-v4": 
    {
        'max_reward': 19.5,
        'hit_reward': 0,
        'frame_skip': 4, # frame skipping introduced by the wrappers
        'running_screen': False # scrolling screen games where the player dies with a zero reward after being hit. scoring is only based on how many target the player hits
    },
    "ALE/Riverraid-v5":
    {
        'max_reward': 2 * 1e5,
        'hit_reward': 60,
        'frame_skip': 4, # frame skipping introduced by the wrappers
        'running_screen': True # scrolling screen games where the player dies with a zero reward after being hit. scoring is only based on how many target the player hits
    },
    "ALE/SpaceInvaders-v5":
    {
        'max_reward': 500,
        'hit_reward': 200, # available rewards are 5, 10, 15, 20, 25, 30, 200 (rare)
        'frame_skip': 4, # frame skipping introduced by the wrappers
        'running_screen': True # scrolling screen games where the player dies with a zero reward after being hit. scoring is only based on how many target the player hits
    },
    "ALE/BeamRider-v5":
    {
        'max_reward': 2000,
        'hit_reward': 15,
        'frame_skip': 4, # frame skipping introduced by the wrappers
        'running_screen': True, # scrolling screen games where the player dies with a zero reward after being hit. scoring is only based on how many target the player hits
        "terminal_on_life_loss": False,
        "update_max_hit": True,
        "total_lives": 3
    }
}

if __name__ == "__main__":
    ENV_NAME = "ALE/BeamRider-v5"
    map_name = ENV_NAME.replace(r'/', '-')
    ROOT_FOLDER =  "runs/" + map_name + '-dqn-per_conv_not_solved/'
    MODEL_FILE = ROOT_FOLDER + map_name + "-best.dat"
    print(games_setup[ENV_NAME].get('terminal_on_life_loss', True))
    env = wrappers.make_env(
        ENV_NAME,
        render_mode="rgb_array",
        frame_skip=games_setup[ENV_NAME]['frame_skip'],
        terminal_on_life_loss=games_setup[ENV_NAME].get('terminal_on_life_loss', True)
        )
    env = gym.wrappers.RecordVideo(env, video_folder=ROOT_FOLDER + 'video_play')
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    weights = torch.load(MODEL_FILE, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(weights)

    state, _ = env.reset()
    total_reward = 0.0

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    env.close()