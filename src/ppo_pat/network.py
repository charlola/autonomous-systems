import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Net = FeedForwardNN
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    # state = observation = obs
    def forward(self, state):

        # convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        activation1 = F.relu(self.layer1(state))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
