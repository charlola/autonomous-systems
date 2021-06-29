import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        weight = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(weight)
        self.eval()

class Net(BaseNet):
    def __init__(self, device, in_dim, hidden_units, out_dim, activation):
        BaseNet.__init__(self)

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
        
        self.net = nn.Sequential(*layers).to(device)
        
class ActorNet(BaseNet):
    def __init__(self, device, in_dim, hidden_units, out_dim, activation):
        BaseNet.__init__(self)

        # get activation function
        activation = getattr(nn, activation)

        units = [in_dim] 
        for unit in hidden_units:
            units += [unit, unit]
        
        layers = list()
        for input_dim, output_dim in zip(*[iter(units)]*2):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation())
        
        self.net = nn.Sequential(*layers).to(device)

        self.mean = nn.Linear(units[-1], out_dim).to(device)
        self.mean.weight.data.fill_(1e-5)
        self.mean.bias.data.fill_(1e-5)
        self.std = nn.Linear(units[-1], out_dim).to(device)
        self.std.weight.data.fill_(1e-5)
        self.std.bias.data.fill_(1e-5)
