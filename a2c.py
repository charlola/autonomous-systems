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
        
        actor_loss, critic_loss, entropy = self.apply(states, actions, log_probs, A_k, rewards, discounted_return)
        
        return actor_loss, critic_loss, entropy
