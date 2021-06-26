import torch
import numpy as np

from actorcritic import ActorCritic, AdvancedActorCritic


class PPO(ActorCritic):
    def __init__(self, args):
        ActorCritic.__init__(self, args)

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
        
        # default at 5 updates per iteration
        for _ in range(self.args.ppo_episodes):
            
            # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
            V, current_log_probs, entropy = self.evaluate(states, actions)

            # Calculate ratios
            ratios = torch.exp(current_log_probs - log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

            # Calculate actor and critic loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.args.noise * entropy
            critic_loss = self.mse(V, discounted_return)

            # Calculate gradients and perform backward propagation
            self.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)
        std_rewads  = np.std(sum_rewards)

        return std_rewads, sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item(), entropy
    
class AdvancedPPO(AdvancedActorCritic):
    def __init__(self, args):
        AdvancedActorCritic.__init__(self, args)
    
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
        
        # default at 5 updates per iteration
        for _ in range(self.args.ppo_episodes):
            
            # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
            V, current_log_probs, entropy = self.evaluate(states, actions)

            # Calculate ratios
            ratios = torch.exp(current_log_probs - log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

            # Calculate actor and critic loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.args.noise * entropy
            critic_loss = self.mse(V, discounted_return)

            # Calculate gradients and perform backward propagation
            self.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)
        std_rewads  = np.std(sum_rewards)

        return std_rewads, sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item(), entropy
