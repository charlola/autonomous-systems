import math
import os

import numpy as np
import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.a2c.agent import Agent
import torch as T


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

        self.fc_value = nn.Linear(nr_hidden_units, 1)

    def forward(self, x):
        base_out = self.fc_base(x)
        return self.fc_mu(base_out), self.fc_sigma(base_out), self.fc_value(base_out)

    def save(self, checkpoint_path):
        T.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        weight = T.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(weight)
        self.eval()


class A2CAgent(Agent):
    def __init__(self, hyperparams, params):
        Agent.__init__(self, hyperparams)
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.transitions = []
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.entropy_factor = hyperparams["entropy_factor"]
        self.advantage = hyperparams["advantage"]
        self.gamma = hyperparams["gamma"]
        self.alpha = hyperparams["alpha"],
        self.a2c_net = A2CNet(
            params["nr_input_features"],
            params["nr_actions"],
            hyperparams["nr_hidden_units"],
        ).to(self.device)

        # load model from file
        if params["load_model"] and os.path.isfile(params["model"]):
            self.a2c_net.load(params["model"])

        self.optimizer = T.optim.Adam(self.a2c_net.parameters(), lr=hyperparams["alpha"])

    def save(self, checkpoint_path):
        self.a2c_net.save(checkpoint_path)

    def policy(self, state):
        mu, sigma, _ = self.a2c_net(T.tensor([state], device=self.device, dtype=T.float32))
        dist = T.distributions.Normal(mu, sigma)
        action = dist.sample()
        action = T.flatten(action)
        action = T.clip(action, -1, 1)
        return action.data.cpu().numpy(), dist.entropy()

    def calculate_discounted_reward(self, rewards):
        discounted_returns = []
        R = 0  # Return
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return discounted_returns

    def advantage_temporal_difference(self, reward, value, next_value):
        return reward + self.gamma * next_value - value

    def advantage_r_l(self, R, value):
        return R - value

    def get_advantage(self, R, reward, value, next_value):
        if self.advantage == "TD": return self.advantage_temporal_difference(reward, value, next_value)
        elif self.advantage == "RL": return self.advantage_r_l(R, value)
        return self.advantage_r_l()

    def calc_policy_loss(self, mu_v, sig_v, action_v, advantage):
        #return -dist.log_prob(action) * advantage
        p1 = -((mu_v - action_v) ** 2) / (2 * sig_v.clamp(min=1e-3))
        p2 = -T.log(torch.sqrt(2 * math.pi * sig_v))
        return (p1 + p2) * advantage

    def calc_entropy_loss(self, sig_v):
        # return dist.entropy() * self.entropy_factor
        return self.entropy_factor * (-(T.log(2*math.pi*sig_v) + 1)/2)

    def calc_value_loss(self, values, reward):
        return T.nn.MSELoss()(values.squeeze(-1), reward)

    def update(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        loss = None
        policy_loss = None
        entropy_loss = None
        value_loss = None
        if done:
            self.optimizer.zero_grad()
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))

            rewards = T.tensor(rewards, device=self.device, dtype=T.float)
            actions = T.tensor(actions, device=self.device, dtype=T.float32)
            discounted_returns = self.calculate_discounted_reward(rewards)
            discounted_returns = T.tensor(discounted_returns, device=self.device, dtype=T.float).detach()
            normalized_returns = (discounted_returns - discounted_returns.mean())
            normalized_returns /= (discounted_returns.std() + self.eps)

            # Calculating probabilities and state_values. Sigma is seen as probability.
            action_mus, action_sigs, state_values = self.a2c_net(T.tensor(states, device=self.device, dtype=T.float32))
            _, _, next_state_values = self.a2c_net(T.tensor(next_states, device=self.device, dtype=T.float32))

            # Calculate normal distributions
            action_distributions = [T.distributions.Normal(action_mu, action_sig) for action_mu, action_sig in zip(action_mus, action_sigs)]

            # Calculate loss
            policy_losses = []
            value_losses = []
            entropy_losses = []

            for mu_v, sig_v, dist, action, value, next_value, R, reward in zip(action_mus, action_sigs, action_distributions, actions, state_values,
                                                                  next_state_values, normalized_returns, rewards):
                advantage = self.get_advantage(R, reward, value, next_value)
                # policy_losses.append(self.calc_policy_loss(dist, action, advantage))
                policy_losses.append(self.calc_policy_loss(mu_v, sig_v, actions, advantage))
                # entropy_losses.append(self.calc_entropy_loss(dist))
                entropy_losses.append(self.calc_entropy_loss(sig_v))
                value_losses.append(self.calc_value_loss(value, reward))

            policy_loss = T.stack(policy_losses).mean()
            entropy_loss = T.stack(entropy_losses).mean()
            value_loss = T.stack(value_losses).mean()
            loss = policy_loss + value_loss + entropy_loss
            loss.backward()

            self.optimizer.step()
            self.transitions.clear()

        return loss, policy_loss, value_loss, entropy_loss