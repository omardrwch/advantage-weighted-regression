import torch
import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    """MLP."""
    def __init__(self,input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activate_last=False):
        super(MLP, self).__init__()
        hidden_sizes = tuple(hidden_sizes) + (output_dim,)
        n_layers = len(hidden_sizes)
        prev_dim = input_dim
        layers = []
        for ii in range(n_layers):
            dim = hidden_sizes[ii]
            linear_layer = nn.Linear(in_features=prev_dim, out_features=dim)
            layers.append(linear_layer)
            if ii < n_layers - 1 or activate_last:
                layers.append(nn.ReLU())
            prev_dim = dim
        self._mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._mlp(x)
