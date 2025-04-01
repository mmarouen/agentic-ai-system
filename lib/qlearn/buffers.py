import numpy as np
from dataclasses import fields, dataclass
import collections
import torch

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    new_state: np.ndarray

    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))

class ExperienceBuffer:
    def __init__(self, capacity, alpha, beta, beta_frames, per_beta_update_frequency, per):
        self.buffer = collections.deque(maxlen=capacity)
        # self.priorities = collections.deque(maxlen=capacity)
        self.priorities = np.empty((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.sum_priorities = 0
        self.beta = beta
        self.alpha = alpha
        # self.beta_rate = (1.0 - beta) / beta_frames
        self.beta_rate = (1.0 - beta) / (beta_frames / per_beta_update_frequency)
        self.capacity = capacity
        self.current_indices = []
        self.epsilon = 1e-5
        self.current_insert = 0
        self.per_beta_update_frequency = per_beta_update_frequency
        self.per = per

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        if self.per:
            if len(self.buffer) == self.capacity:
                old_priority = self.priorities[0]
                self.sum_priorities = self.sum_priorities - old_priority + self.max_priority
                self.priorities[:-1] = self.priorities[1:]
                self.priorities[-1] = self.max_priority
            else:
                self.sum_priorities += self.max_priority
                # self.current_insert += 1
            # self.priorities.append(self.max_priority)
                self.priorities[len(self.buffer)] = self.max_priority
            
        self.buffer.append(experience)

    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_rate)

    def update_priorities(self, losses):
        losses_alpha = (np.abs(losses) + self.epsilon) ** self.alpha
        for i, loss_ in enumerate(losses_alpha):
            experience_index = self.current_indices[i]
            self.sum_priorities = self.sum_priorities + loss_ - self.priorities[experience_index]
            self.priorities[experience_index] = loss_
            if self.max_priority < loss_:
                self.max_priority = loss_
        self.current_indices = []

    def sample(self, batch_size, device, debug_mode=False):
        current_indices = []
        if self.per:
            # probs = [p / self.sum_priorities for p in self.priorities]
            probs = self.priorities[:len(self.buffer)] / self.sum_priorities
            if not debug_mode:
                self.current_indices = np.random.choice(len(self.buffer), batch_size, replace=True, p=probs)
                current_indices = self.current_indices
            else:
                current_indices = np.random.choice(len(self.buffer), batch_size, replace=True, p=probs)
        else:
            self.current_indices = np.random.choice(len(self.buffer), batch_size, replace=True)
            current_indices = self.current_indices
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in current_indices])
        
        if self.per and not debug_mode:
            priorities = [self.priorities[idx] for idx in self.current_indices]
        else:
            priorities = [0] * len(states)

        return (
            torch.as_tensor(np.asarray(states)).to(device),
            torch.tensor(actions, device=device, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.ByteTensor(dones).bool().to(device),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.tensor(priorities, dtype=torch.float32, device=device)
        )
