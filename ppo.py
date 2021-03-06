import torch
import numpy as np

from actorcritic import ActorCritic


class PPO(ActorCritic):
    def __init__(self, args):
        ActorCritic.__init__(self, args)

        self.model = self.create_model()
        
        if args.mini_batch_size <= 0 or args.mini_batch_size > args.batch_size:
            args.mini_batch_size = args.batch_size

        self.minibatches = int(self.args.batch_size / self.args.mini_batch_size)
        self.batch_indices = np.arange(self.args.batch_size)


    def get_actor_loss(self, current_log_probs, log_probs, A, entropy):
        # Calculate ratios
        ratios = torch.exp(current_log_probs - log_probs)

        # Calculate surrogate losses
        surr1 = ratios * A
        surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A

        # Calculate actor and critic loss
        actor_loss = -torch.min(surr1, surr2).mean() - self.get_noise(entropy)

        return actor_loss

    def get_critic_loss(self, V, rewards, discounted_return):
        return self.mse(V, discounted_return)

    def learn(self, states, next_states, actions, log_probs, rewards, dones, discounted_return):

        # Calculate Advantage
        A = self.get_advantage(states, next_states, actions, dones, discounted_return)
        
        # default at 5 updates per iteration
        for _ in range(self.args.ppo_episodes):

            if self.args.shuffle_mini_batch:
                # Shuffle batch indices
                np.random.shuffle(self.batch_indices)

            # Iterate over minibatches
            for minibatch in range(self.minibatches):
                
                # Pick indices for minibatch
                i_start = minibatch       * self.args.mini_batch_size
                i_stop  = (minibatch + 1) * self.args.mini_batch_size
                indices = self.batch_indices[i_start: i_stop]

                # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
                V, current_log_probs, entropy = self.model.evaluate(states[indices], actions[indices])
                
                # Calculate actor and critic loss
                actor_loss  = self.get_actor_loss(current_log_probs, log_probs[indices], A[indices], entropy)
                critic_loss = self.get_critic_loss(V, rewards[indices], discounted_return[indices]) 

                # Calculate gradients and perform backward propagation
                self.model.optimize(actor_loss, critic_loss)


        return actor_loss, critic_loss, entropy