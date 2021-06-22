import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Net = FeedForwardNN
class Net(nn.Module):
    def __init__(self, in_dim, hidden_units, out_dim, activation):
        super(Net, self).__init__()

        # get activation function
        activation = getattr(nn, activation)

        units = [in_dim] 
        for unit in hidden_units:
            units += [unit, unit]
        units += [out_dim]

        layers = list()
        for i, (input_dim, output_dim) in enumerate(zip(*[iter(units)]*2)):
            if i != 0:
                layers.append(activation())
            layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    # state = observation = obs
    def forward(self, state):

        # convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        return self.net(state)
