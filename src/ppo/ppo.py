from src.agent.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import random
from src.ppo.config import params, hyperparams
from src.ppo.actorcritic import ActorCritic

class PPOAgent(Agent):

    def __init__(self, params):
        Agent.__init__(self, params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.alpha_actor = params["alpha_actor"]
        self.alpha_critic = params["alpha_critic"]
        self.gamma = params["gamma"]
        self.step_size = params["step_size"]
        self.k = params["k"]
        self.eps = np.finfo(np.float32).eps.item()

        self.net = ActorCritic(device=self.device)

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
        transition = (state, action, reward, next_state, done)
        self.transitions.append(transition)

        if done or len(self.transitions) == self.step_size:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)

            # Calculate and normalize discounted returns
            discounted_returns = []
            R = 0
            for reward in reversed(rewards):
                R = reward + self.gamma * R
                discounted_returns.insert(0, R)

            discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float).detach()
            normalized_returns = (discounted_returns - discounted_returns.mean())
            normalized_returns /= (discounted_returns.std() + self.eps)

            for _ in range(self.k):
                pass

            actions, action_logprobs = self.predict_policy(states)
            next_action, next_action_logprob = self.predict_policy(next_states)


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
