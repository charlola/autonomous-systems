from actorcritic import ActorCritic
import numpy as np

class A2C(ActorCritic):
    def __init__(self, args):
        ActorCritic.__init__(self, args)

    def get_actor_loss(self, current_log_probs, log_probs, A_k, entropy):
        return (-current_log_probs * A_k).sum() - self.args.noise * entropy

    def get_critic_loss(self, V, rewards, discounted_return):
        return self.mse(V, discounted_return)

    def learn(self, states, next_states, actions, log_probs, rewards, dones, discounted_return):

        # Calculate Advantage
        A_k = self.get_advantage(states, next_states, actions, dones, discounted_return)
        
        # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
        V, current_log_probs, entropy = self.model.evaluate(states, actions)
    
        # Calculate actor and critic loss
        actor_loss  = self.get_actor_loss(current_log_probs, log_probs, A_k, entropy)
        critic_loss = self.get_critic_loss(V, rewards, discounted_return) 

        # Calculate gradients and perform backward propagation
        self.model.optimize(actor_loss, critic_loss)
        
        return actor_loss.item(), critic_loss.item(), entropy
