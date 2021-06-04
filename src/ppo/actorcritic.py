from torch._C import device
from src.agent.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import random
from src.ppo.config import params, hyperparams

class ActorCritic(nn.Module):
    def __init__(self, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_dim = params['action_dim']

        self.set_action_std(hyperparams['action_std_init'])
        
        self.actor = nn.Sequential(
            nn.Linear(params['state_dim'], hyperparams['nr_hidden_units']),
            nn.Tanh(), 
            nn.Linear(hyperparams['nr_hidden_units'], hyperparams['nr_hidden_units']),
            nn.Tanh(),
            nn.Linear(hyperparams['nr_hidden_units'], params['action_dim']),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(params['state_dim'], hyperparams['nr_hidden_units']),
            nn.Tanh(),
            nn.Linear(hyperparams['nr_hidden_units'], hyperparams['nr_hidden_units']),
            nn.Tanh(),
            nn.Linear(hyperparams['nr_hidden_units'], 1),
            nn.Tanh(),
        )

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def set_action_std(self, action_std):
        dim   = (self.action_dim,)
        value = action_std * action_std
        self.action_var = torch.full(dim, value).to(self.device)
