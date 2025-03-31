import cv2
import gymnasium as gym
import numpy as np
import collections
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, info = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _, info = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ActionMappingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # For Pong, we keep only actions: NOOP (0), UP (2), DOWN (3)
        self._action_map = {0: 0, 1: 2, 2: 3}
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def action(self, action):
        return self._action_map[action]
'''
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert 'FIRE' in env.unwrapped.get_action_meanings(), "Environment does not have FIRE action"
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, info = self.env.step(1)  # send FIRE
        if done:
            self.env.reset(**kwargs)
        #obs, _, done, _, info = self.env.step(2)
        #if done:
        #    self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


def make_env(env_name, frame_skip_count, frame_stacking_count):
    env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip_count,
        grayscale_obs=True,
        scale_obs=True,
        noop_max=5
        )
    env = FireResetEnv(env)
    env = ActionMappingWrapper(env)
    new_observation_space = gym.spaces.Box(
        low=0, high=1, shape=(env.observation_space.shape[0], env.observation_space.shape[1]), dtype=np.float32
    )
    env = TransformObservation(env, lambda obs: np.moveaxis(obs, -1, 0), new_observation_space)  # Convert to PyTorch format
    env = FrameStackObservation(env, stack_size=frame_stacking_count)  # Use built-in frame stacking
    return env

'''
 
def make_env(env_name, frame_skip_count, frame_stacking_count):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env, skip=frame_skip_count)
    env = FireResetEnv(env)
    # env = ActionMappingWrapper(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, frame_stacking_count)
    return ScaledFloatFrame(env)
