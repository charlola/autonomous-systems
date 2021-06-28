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
    def get_actor_loss(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_critic_loss(self, V, rewards, discounted_return):
        raise NotImplementedError

    def get_advantage(self, states, actions, discounted_return):
        # Evaluate state and actions
        V, _, _ = self.model.evaluate(states, actions)

        # Calculate Advantage
        A_k = discounted_return - V.detach()

        if self.args.normalize:
            # Normalize Advantages (Trick: makes PPO more stable)
            # Subtracting 1e-10, so there will be no possibility of dividing by 0
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        return A_k

    def learn(self):
        # Perform rollout to get batches
        states, actions, log_probs, rewards, sum_rewards, discounted_return = self.rollout()

        # Calculate Advantage
        A_k = self.get_advantage(states, actions, discounted_return)
        
        # default at 5 updates per iteration
        for _ in range(self.args.ppo_episodes):
            
            # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
            V, current_log_probs, entropy = self.model.evaluate(states, actions)

            # Calculate actor and critic loss
            actor_loss  = self.get_actor_loss(current_log_probs, log_probs, A_k, entropy)
            critic_loss = self.get_critic_loss(V, rewards, discounted_return) 

            # Calculate gradients and perform backward propagation
            self.model.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)
        std_rewads  = np.std(sum_rewards)

        return std_rewads, sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item(), entropy
    