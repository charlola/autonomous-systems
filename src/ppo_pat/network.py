import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Net = FeedForwardNN
class Net(nn.Module):
    def __init__(self, in_dim, hidden_units, out_dim):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_dim)
        )

    # state = observation = obs
    def forward(self, state):

        # convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        return self.net(state)
