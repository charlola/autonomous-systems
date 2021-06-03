from src.agent.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import random

class ActorCritic(nn.Module):
    def __init__(self, ac_config):
        super(ActorCritic, self).__init__()

        self.action_dim = ac_config['action_dim']
        self.action_var = torch.full((ac_config['action_dim']),ac_config['action_std_init'] * ac_config['action_std_init']).to(ac_config['device'])

        self.actor = nn.Sequential(
            nn.Linear(ac_config['nr_input_features'], ac_config['nr_hidden_units']),
            nn.Tanh(), # evtl nn.Tanh() statt ReLU
            nn.Linear(ac_config['nr_hidden_units'], ac_config['nr_hidden_units']),
            nn.Tanh(),# evtl nn.Tanh() statt ReLU,
            nn.Linear(ac_config['nr_hidden_units'], ac_config['action_dim']),
            nn.Tanh(),# evtl nn.Tanh() statt ReLU,
        )

        self.critic = nn.Sequential(
            nn.Linear(ac_config['nr_input_features'], ac_config['nr_hidden_units']),
            nn.Tanh(), # evtl nn.Tanh() statt ReLU
            nn.Linear(ac_config['nr_hidden_units'], ac_config['nr_hidden_units']),
            nn.Tanh(),# evtl nn.Tanh() statt ReLU,
            nn.Linear(ac_config['nr_hidden_units'], 1),
            nn.Tanh(),# evtl nn.Tanh() statt ReLU,
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
        x = self.fc_net(x)
        x = x.view(x.size(0), -1)
        return F.softmax(self.action_head(x), dim=-1)

class ReplayBuffer:
    def __init__(self, batch_size):
        self.transitions = list()
        self.batch_size = batch_size

    def save(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.size:
            self.transitions.pop(0)

    def update(self, batch_indices, minibatch, deltas):
        for index, transition, delta in zip(batch_indices, minibatch, deltas):
            self.transitions[index] = transition[:5] + (delta,)

    def sample_batch(self, minibatch_size):
        nr_episodes = len(self.transitions)
        if nr_episodes > minibatch_size:
            deltas = np.array([x[5] for x in self.transitions])
            indices = random.choices(range(len(self.transitions)), k=minibatch_size, weights=deltas)
            minibatch = [self.transitions[i] for i in indices]
            return indices, minibatch
        return range(len(self.transitions)), self.transitions

    def clear(self):
        self.transitions.clear()

    def size(self):
        return len(self.transitions)

class PPOAgent(Agent):

    def __init__(self, params):
        Agent.__init__(self, params)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.alpha_actor = params["alpha_actor"]
        self.alpha_critic = params["alpha_critic"]
        self.gamma = params["gamma"]
        self.epsilon = np.finfo(np.float32).eps.item()
        self.k = 5

        self.net = ActorCritic(57, 9)

        optimizer_params = [
            {'params': self.net.actor.parameters(),  'lr': self.alpha_actor},
            {'params': self.net.critic.parameters(), 'lr': self.alpha_critic}
        ]
        self.optimizer = torch.optim.Adam(optimizer_params)

        self.transitions = list()

    def policy(self, state):
        #return np.random.rand(9,1)
        action, action_logprob = self.predict_policy([state])
        return action.detach().cpu().numpy().flatten()

    """
     Predicts the action probabilities.
    """
    def predict_policy(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.net.act(states)

    def update(self, state, action, reward, next_state, done):

        return
        policy_losses = []
        value_losses = []

        for probs, action, value, next_value, R in zip(action_probs, actions, state_values, next_state_values, normalized_returns):
            # reinforce
            advantage = R
            # actor-critic
            advantage = R - value.item()
            # temporal difference actor-critic
            advantage = R + next_value.item() - value.item()

            if advantage >= 0:
                x = (1+self.epsilon) * advantage
            else:
                x = (1-self.epsilon) * advantage


            m = Categorical(probs)
            policy_losses.append(-m.log_prob(action) * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # Optimize joint loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
