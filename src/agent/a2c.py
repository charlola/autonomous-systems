import os

import numpy.random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.agent.agent import Agent
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
        action = T.clip(action, -0.99, 1)
        return action.data.cpu().numpy(), dist

    def calculate_discounted_reward(self, rewards):
        discounted_returns = []
        R = 0  # Return
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return discounted_returns

    def advantage_temporal_difference(self, reward, value, next_value):
        return reward.item() + self.gamma * next_value.item() - value.item()

    def advantage_r_l(self, R, value):
        return R - value

    def get_advantage(self, R, reward, value, next_value):
        if self.advantage == "TD": return self.advantage_temporal_difference(reward, value, next_value)
        elif self.advantage == "RL": return self.advantage_r_l(R, value)
        return self.advantage_r_l()

    def update(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        loss = None
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

            # Calculate losses
            policy_losses = []
            value_losses = []
            entropy_losses = []
            for dist, action, value, next_value, R, reward in zip(action_distributions, actions, state_values,
                                                                   next_state_values, normalized_returns, rewards):
                advantage = self.get_advantage(R, reward, value, next_value)
                policy_losses.append(-dist.log_prob(action) * advantage)
                entropy_losses.append(dist.entropy() * self.entropy_factor)
                value_losses.append(F.smooth_l1_loss(T.tensor(value, device=self.device, dtype=T.float32),
                                                     T.tensor(R, device=self.device, dtype=T.float32)))

            loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum() + T.stack(entropy_losses).sum()
            loss.backward()
            self.optimizer.step()
            self.transitions.clear()

        return loss