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

        self.fc_net_actor = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )

        self.fc_net_critic = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(nr_hidden_units, nr_actions)
        self.critic_head = nn.Linear(nr_hidden_units, 1)

    def forward(self, x):
        # actor
        x_act = self.fc_net_actor(x)
        x_act = x_act.view(-1, x_act.size(0))

        # critic
        x_cri = self.fc_net_critic(x)
        x_cri = x_cri.view(-1, x_cri.size(0))

        return F.softmax(self.actor_head(x_act), dim=-1), self.critic_head(x_cri)


class A2CAgent(Agent):

    def policy(self, state):

        return np.random.rand(1, 9)

    def update(self, state, action, reward, next_state, done):
        pass

    def __init__(self, params):
        Agent.__init__(self, params)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
