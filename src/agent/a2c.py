import numpy.random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.agent.agent import Agent
import torch as T
import numpy as np

class A2CNet(nn.Module):
    # Todo use model = MyModel(); model.cuda()
    def __init__(self, nr_input_features, nr_actions, nr_hidden_units):
        super(A2CNet, self).__init__()

        self.fc_base = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        )

    def forward(self, x):
        base_out = self.fc_base(x)
        return self.fc_mu(base_out), self.fc_sigma(base_out)

class A2CAgent(Agent):
    def __init__(self, params):
        Agent.__init__(self, params)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.gamma = params["gamma"]
        self.alpha = params["alpha"],
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

    def policy(self, state):
        # Todo moving back to cpu?
        mu, sigma = self.actor_net(T.tensor([state], device=self.device, dtype=T.float32))
        mu = mu.data.cpu().numpy()
        sigma = sigma.data.cpu().numpy()
        actions = numpy.random.normal(mu, sigma)
        actions = numpy.clip(actions, -1, 1)
        return  actions



    def update(self, state, action, reward, next_state, done):
        pass


