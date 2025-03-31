import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium import Wrapper
import collections
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common import atari_wrappers


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_space.shape, dtype=np.float32
        )
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.n_steps = n_steps
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        obs, extra = self.env.reset()
        for _ in range(self.buffer.maxlen-1):
            #self.buffer.append(self.env.observation_space.low)
            self.buffer.append(obs)
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)

def make_atari_env(env_name: str, eval_mode:bool=False, render_mode=None, frame_skip: int=4, tll: bool=True, frame_stacking: int=4, **kwargs):
    env = gym.make(env_name, frameskip=1, render_mode=render_mode, **kwargs)
    env = atari_wrappers.AtariWrapper(
        env,
        clip_reward=False,
        noop_max=30,
        frame_skip=frame_skip,
        terminal_on_life_loss=False if eval_mode else tll
        )
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=frame_stacking)
    env = ScaledFloatFrame(env)
    return env

def make_env(env_id: int, is_atari: bool, record_video_freq: int, env_name: str, video_root: str, eval_mode:bool=False, frame_skip: int=4, tll: bool=True, frame_stacking: int=4, **kwargs):
    def dummy():
        render_mode = "rgb_array" if record_video_freq > 0 and env_id == 0 else None
        if is_atari:
            env = make_atari_env(
                env_name=env_name,
                frame_skip=frame_skip,
                tll=tll,
                frame_stacking=frame_stacking,
                render_mode=render_mode,
                eval_mode=eval_mode,
                **kwargs)
        else:
            env = gym.make(env_name, render_mode=render_mode, **kwargs)
        if env_id == 0 and record_video_freq > 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_root,
                name_prefix='train',
                step_trigger=lambda x: x % record_video_freq == 0)
        return env
    return dummy