import numpy as np
import gymnasium as gym

class ObsNormalize():
    def __init__(self, state_dim: int):
        self.mean = np.zeros(state_dim)
        self.sq_sum = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.min = np.zeros(state_dim)
        self.max = np.zeros(state_dim)
        self.n_obs = 0
        self.epsilon = 1e-8
        self.is_init = True
        self.mean_ = np.zeros(state_dim)
        self.var_ = np.ones(state_dim)
        self.std_ = np.ones(state_dim)

    def unfreeze(self):
        self.mean_ = self.mean
        self.var_ = self.var
        self.std_ = np.sqrt(self.var_)

    def _rolling_update(self, batch_sum, batch_sq, batch_min, batch_max, p):
        if self.is_init:
            self.min = batch_min
            self.max = batch_max
        else:
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)
        total_obs = self.n_obs + p
        batch_mean = batch_sum / p
        batch_var = batch_sq / p - batch_mean ** 2
        delta = batch_mean - self.mean
        self.sq_sum = self.sq_sum + batch_var * p + self.n_obs * p * delta ** 2 / total_obs
        self.mean = (self.n_obs * self.mean + batch_mean * p) / total_obs
        self.n_obs = total_obs
        self.var = self.sq_sum / (self.n_obs - 1)
        self.is_init = False

    def update(self, batch):
        batch_sum = np.sum(batch, axis=0)
        batch_sq = np.sum(batch ** 2, axis=0)
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        size = len(batch)
        self._rolling_update(batch_sum, batch_sq, batch_min, batch_max, size)

    def get_stats(self):
        return self.mean_, np.sqrt(self.var_ + self.epsilon)

    def normalize(self, observation):
        return (observation - self.mean_) / (self.std_ + self.epsilon)

    def warmup(self, test_env: gym.Env):
        observations = []
        for _ in range(100):
            current_state, _ = test_env.reset()
            observations.append(current_state)
            while True:
                action = test_env.action_space.sample()
                new_obs, _, done, truncated, *_ = test_env.step(action)
                observations.append(new_obs)
                if done or truncated:
                    break
        test_env.close()
        self.update(np.asarray(observations))
