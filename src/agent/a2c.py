import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.agent.agent import Agent
import torch
import numpy as np

class A2CNet(nn.Module):
    # Todo use model = MyModel(); model.cuda()
    def __init__(self, nr_input_features, nr_actions, nr_hidden_units):
        super(A2CNet, self).__init__()

        self.fc_1 = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.fc_head = nn.Linear(nr_hidden_units, nr_actions)

    def forward(self, x):
        x = self.fc_1(x)
        x = x.view(x.size(0), -1)
        return self.fc_head(x)

class A2CAgent(Agent):

    def policy(self, state):
        # Todo moving back to cpu?
        actions = self.actor_net(torch.tensor([state], device=self.device, dtype=torch.float32)).cpu().detach()
        return actions[0]

    def update(self, state, action, reward, next_state, done):
        pass

    def __init__(self, params):
        Agent.__init__(self, params)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor_net = A2CNet(
            params["nr_input_features"],
            params["nr_actions"],
            params["nr_hidden_units"],
        ).to(self.device)
        self.critic_net = A2CNet(
            params["nr_input_features"],
            1,
            params["nr_hidden_units"],
        ).to(self.device)
