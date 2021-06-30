from abc import abstractmethod

import torch
import numpy as np

from agent import Agent
from model import Model, AdvancedModel


class ActorCritic(Agent):
    def __init__(self, args):
        Agent.__init__(self, args)

    def create_model(self):
        if self.args.algorithm in ["appo", "aa2c"]:
            model = AdvancedModel(self.args)
        else:
            model = Model(self.args)
        return model
    
    @abstractmethod
    def learn(self):
        raise NotImplementedError

    def get_noise(self, entropy):
        # Define noise factor for entropy 'loss'
        noise = self.args.noise
        if self.args.noise_decay == "linear":
            noise *= (1 - self.args.episode / self.args.episodes)
        elif self.args.noise_decay == "geometric":
            noise *= 0.9999 ** self.args.episode
        return noise * entropy

    def get_advantage(self, states, next_states, actions, dones, discounted_return):
        # Evaluate state and actions
        V, _, _ = self.model.evaluate(states, actions)

        has_next = torch.logical_xor(dones, torch.ones_like(dones, device=self.args.device)) 

        # Calculate Advantage ( reinforce | temporal | advantage )
        if self.args.advantage == "reinforce":
            A_k = discounted_return
        elif self.args.advantage == "temporal":
            V_next, _, _ = self.model.evaluate(next_states, actions)
            A_k = discounted_return + V_next.detach() * self.args.gamma * has_next - V.detach()
        else:
            A_k = discounted_return - V.detach()

        if self.args.gae_lambda > 0:
            next_advantage = 0
            for i in reversed(range(len(A_k))):
                A_k[i] = A_k[i] + self.args.gamma * self.args.gae_lambda * next_advantage * has_next[i]
                next_advantage = A_k[i] 

        if self.args.normalize == "advantage":
            # Normalize Advantages (Trick: makes PPO more stable)
            # Subtracting 1e-10, so there will be no possibility of dividing by 0
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        return A_k