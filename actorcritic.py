from abc import abstractmethod

import torch
import numpy as np

from agent import Agent
from model import Model, AdvancedModel


class ActorCritic(Agent):
    def __init__(self, args):
        if args.algorithm in ["appo", "aa2c"]:
            model = AdvancedModel(args)
        else:
            model = Model(args)
        
        Agent.__init__(self, args, model)
    
    @abstractmethod
    def learn(self):
        raise NotImplementedError        

    def apply(self, states, actions, log_probs, A_k, rewards, discounted_return):
        # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
        V, current_log_probs, entropy = self.model.evaluate(states, actions)
        
        # Calculate actor and critic loss
        actor_loss  = self.get_actor_loss(current_log_probs, log_probs, A_k, entropy)
        critic_loss = self.get_critic_loss(V, rewards, discounted_return) 

        # Calculate gradients and perform backward propagation
        self.model.optimize(actor_loss, critic_loss)

        return actor_loss.item(), critic_loss.item(), entropy

    def get_advantage(self, states, next_states, actions, dones, discounted_return):
        # Evaluate state and actions
        V, _, _ = self.model.evaluate(states, actions)

        # reinforce | temporal | advantage

        # Calculate Advantage
        if self.args.advantage == "reinforce":
            A_k = discounted_return
        elif self.args.advantage == "temporal":
            has_next = torch.logical_xor(dones, torch.ones_like(dones, device=self.args.device)) 
            V_next, _, _ = self.model.evaluate(next_states, actions)
            A_k = discounted_return + V_next.detach() * self.args.gamma * has_next - V.detach()
        else:
            A_k = discounted_return - V.detach()

        if self.args.normalize == "advantage":
            # Normalize Advantages (Trick: makes PPO more stable)
            # Subtracting 1e-10, so there will be no possibility of dividing by 0
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        return A_k