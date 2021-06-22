import torch
import numpy as np
from torch.distributions import MultivariateNormal

from network import Net
from agent import Agent

class PPO(Agent):
    def __init__(self, args):
        Agent.__init__(self, args)
        
        # Initialize actor and critic networks
        self.actor  = Net(args.state_dim, args.hidden_units, args.act_dim)
        self.critic = Net(args.state_dim, args.hidden_units, 1)

        # Initialize optimizer
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Create our variable for the matrix
        # Chose 0.5 for standarddeviation
        # Create covariance matrix
        self.cov_var = torch.full(size=(args.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, state):

        # Query actor network for mean action
        # Same as calling self.actor.forward(state)
        mean = self.actor(state)

        # Creating Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample action from distribution and get log prob
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        # clip action to action space bounds
        action = torch.clip(action, min=self.args.action_low, max=self.args.action_high)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine.
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, states, actions):

        # Query critic network for a value V for each state in batch_state
        V = self.critic(states).squeeze()

        # Calculate log probabilites of batch actions using most recent actor network
        mean = self.actor(states)

        # Creating Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()

        return V, log_probs, entropy

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
        for _ in range(self.args.k):

            # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
            V, current_log_probs, entropy = self.evaluate(states, actions)

            # Calculate ratios
            ratios = torch.exp(current_log_probs - log_probs)

            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

            # Calculate actor and critic loss
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = torch.nn.MSELoss()(V, discounted_return)

            # Calculate gradients and perform backward propagation
            self.optimize(actor_loss, critic_loss)
        
        avg_rewards = np.mean(sum_rewards)

        return sum_rewards, avg_rewards, actor_loss.item(), critic_loss.item()
    
    def optimize(self, actor_loss, critic_loss):
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()