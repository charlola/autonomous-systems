import torch
import numpy as np

from actorcritic import ActorCritic


class PPO(ActorCritic):
    def __init__(self, args):
        ActorCritic.__init__(self, args)

    def get_actor_loss(self, current_log_probs, log_probs, A_k, entropy):
        # Calculate ratios
        ratios = torch.exp(current_log_probs - log_probs)

        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

        # Calculate actor and critic loss
        actor_loss = -torch.min(surr1, surr2).mean() - self.args.noise * entropy

        return actor_loss

    def get_critic_loss(self, V, rewards, discounted_return):
        return self.mse(V, discounted_return)

    def apply(self, states, actions, log_probs, A_k, rewards, discounted_return):
        # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
        V, current_log_probs, entropy = self.model.evaluate(states, actions)
        
        # Calculate actor and critic loss
        actor_loss  = self.get_actor_loss(current_log_probs, log_probs, A_k, entropy)
        critic_loss = self.get_critic_loss(V, rewards, discounted_return) 

        # Calculate gradients and perform backward propagation
        self.model.optimize(actor_loss, critic_loss)

        return actor_loss.item(), critic_loss.item(), entropy

    def learn(self, states, next_states, actions, log_probs, rewards, dones, discounted_return):

        # Calculate Advantage
        A_k = self.get_advantage(states, next_states, actions, dones, discounted_return)
        
        # default at 5 updates per iteration
        for _ in range(self.args.ppo_episodes):
            
            if self.args.mini_batch_size > 0:
                # default at batch_size 32 per iteration
                for minibatch in range(int(self.args.batch_size / self.args.mini_batch_size)):

                    # create indices
                    indices = np.arange(minibatch * self.args.mini_batch_size, (minibatch + 1) * self.args.mini_batch_size)

                    actor_loss, critic_loss, entropy = self.apply(states[indices], actions[indices], log_probs[indices], A_k[indices], rewards[indices], discounted_return[indices])
            else:
                actor_loss, critic_loss, entropy = self.apply(states, actions, log_probs, A_k, rewards, discounted_return)
        

        return actor_loss, critic_loss, entropy