import torch
import numpy as np

from ppo import PPO, AdvancedPPO

class A2C(PPO):
    def __init__(self, args):
        PPO.__init__(self, args)

    def learn(self):

        # Perform rollout to get batches
        states, actions, log_probs, rewards, sum_rewards, discounted_return = self.rollout()

        # Evaluate state and actions
        V, _, entropy = self.evaluate(states, actions)

        # Calculate Advantage
        A_k = discounted_return - V.detach()

        # Normalize Advantages (Trick: makes PPO more stable)
        # Subtracting 1e-10, so there will be no possibility of dividing by 0
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        
        V, current_log_probs, entropy = self.evaluate(states, actions) 

        actor_loss = (-current_log_probs * A_k).mean()
        critic_loss = self.mse(V, discounted_return)
        
        # Calculate gradients and perform backward propagation
        self.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)
        std_rewads  = np.std(sum_rewards)

        return std_rewads, sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item(), entropy


class AdvancedA2C(AdvancedPPO):
    def __init__(self, args):
        AdvancedPPO.__init__(self, args)

    def learn(self):

        # Perform rollout to get batches
        states, actions, log_probs, rewards, sum_rewards, discounted_return = self.rollout()

        # Evaluate state and actions
        V, _, entropy = self.evaluate(states, actions)

        # Calculate Advantage
        A_k = discounted_return - V.detach()

        # Normalize Advantages (Trick: makes PPO more stable)
        # Subtracting 1e-10, so there will be no possibility of dividing by 0
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        
        V, current_log_probs, entropy = self.evaluate(states, actions) 

        actor_loss = (-current_log_probs * A_k).mean()
        critic_loss = self.mse(V, discounted_return)
        
        # Calculate gradients and perform backward propagation
        self.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)
        std_rewads  = np.std(sum_rewards)

        return std_rewads, sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item(), entropy