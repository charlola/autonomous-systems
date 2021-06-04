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

    def forward(self, x):
        base_out = self.fc_base(x)
        return self.fc_mu(base_out), self.fc_sigma(base_out)

class A2CAgent(Agent):
    def __init__(self, params):
        Agent.__init__(self, params)
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.transitions = []
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
        self.optimizer = T.optim.Adam(self.actor_net.parameters(), lr=params["alpha"])

    def policy(self, state):
        mu, sigma = self.actor_net(T.tensor([state], device=self.device, dtype=T.float32))
        action = T.distributions.Normal(mu, sigma).sample()
        action = T.flatten(action)
        # Todo add lower bound upper bound for action space
        action = numpy.clip(action, -1, 1)
        return action

    def calculate_discounted_reward(self, rewards):
        discounted_returns = []
        R = 0 # Return
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return discounted_returns

    def advantage_temporal_difference(self, reward, value, next_value):
        return reward.item() + self.gamma * next_value.item() - value.item()

    def advantage(self, R, value):
        return R - value

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
            _ , action_probs = self.actor_net(T.tensor(states, device=self.device, dtype=T.float32))
            _, next_action_probs = self.actor_net(T.tensor(next_states, device=self.device, dtype=T.float32))
            state_values, _ = self.critic_net(T.tensor(states, device=self.device, dtype=T.float32))
            next_state_values, _ = self.critic_net(T.tensor(next_states, device=self.device, dtype=T.float32))

            # Calculate losses
            policy_losses = []
            value_losses = []
            for probs, action, value, next_value, R, reward in zip(action_probs, actions, state_values, next_state_values, normalized_returns, rewards):
                action = T.flatten(action)
                probs = T.flatten(probs)
                #advantage = self.advantage_temporal_difference(reward, value, next_value)
                advantage = self.advantage(R, value)
                m = Categorical(probs)
                policy_losses.append(-m.log_prob(action) * advantage)
                value_losses.append(F.smooth_l1_loss(T.tensor(value, device=self.device, dtype=T.float32), T.tensor(R, device=self.device, dtype=T.float32)))

            loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum()
            loss.backward()
            self.optimizer.step()
            self.transitions.clear()

        return loss


