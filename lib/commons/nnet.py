import numpy as np
import torch
import torch.nn as nn

class CoreNet(nn.Module):
    def __init__(self, state_dim, action_dim, is_atari: bool, is_discrete: bool = False, hidden_states: int = 64):
        super(CoreNet, self).__init__()
        self.is_atari = is_atari
        self.is_discrete = is_discrete
        self.action_dim = action_dim
        if is_atari:
            self.conv = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),

                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            for layer in self.conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            conv_out_size = self._get_conv_out(state_dim)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(state_dim[0], hidden_states),
                nn.ReLU(),
                nn.LayerNorm(hidden_states),
                nn.Linear(hidden_states, hidden_states),
                nn.ReLU(),
                nn.LayerNorm(hidden_states),
                nn.Linear(hidden_states, hidden_states),
                nn.ReLU(),
            )
        self.hidden_state_fc = 512 if is_atari else hidden_states
        self.value = nn.Linear(self.hidden_state_fc, 1)
        self.init_linear_layers()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if self.is_atari:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
        return self.fc(x)

    def init_linear_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                gain = np.sqrt(2)
                if layer.out_features == 1:
                    gain = 1.
                elif layer.out_features == self.action_dim:
                    gain = 0.01
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def get_value(self, x: torch.Tensor):
        return self.value(self.forward(x)).squeeze(-1)
