import torch
import numpy as np
from src.ppo_pat.network import Net
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self, env):

        # Default varlues for hyperparameters
        self.timesteps_per_batch = 4800         # 4800
        self.max_timesteps_per_episode = 1600   # 1600
        self.gamma = 0.95                       # 0.95
        self.n_updates_per_iteration = 5        # 5
        self.clip = 0.2                         # 0.2
        self.lr = 0.005                         # 0.005
        self.hidden_layer = 64                  # 64

        # Environment information
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor  = Net(self.state_dim, self.hidden_layer, self.act_dim)
        self.critic = Net(self.state_dim, self.hidden_layer, 1)

        # Create our variable for the matrix
        # Chose 0.5 for standarddeviation
        # Create covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Initialize optimizer
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.logging = {name:[] for name in ["Actor Loss", "Critic Loss"]}

    def learn(self, total_timesteps):

        # Timesteps simulated so far
        self.t_so_far = 0

        episode = 0

        # Collect rewards
        rewards2go = []

        while self.t_so_far < total_timesteps:

            # Perform rollout to get batches
            batch_state, batch_acts, batch_log_probs, batch_rewards_togo, batch_lengths, sum_rewards = self.rollout()

            # Sum up rewards and add to list
            rewards2go += sum_rewards

            # Calculate how many timesteps were collected this batch
            self.t_so_far += np.sum(batch_lengths)

            episode += 1
            
            # Evaluate state and actions
            V, _, entropy = self.evaluate(batch_state, batch_acts)

            # Calculate Advantage
            A_k = batch_rewards_togo - V.detach()

            # Normalize Advantages (Trick: makes PPO more stable)
            # Subtracting 1e-10, so there will be no possibility of dividing by 0
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            algorithm = "ppo"
            if algorithm == "ppo":
                # default at 5 updates per iteration
                for _ in range(self.n_updates_per_iteration):

                    # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
                    V, current_log_probs, entropy = self.evaluate(batch_state, batch_acts)

                    # Calculate ratios
                    ratios = torch.exp(current_log_probs - batch_log_probs)

                    # Calculate surrogate losses
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                    # Calculate actor and critic loss
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = torch.nn.MSELoss()(V, batch_rewards_togo)

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
    
            elif algorithm == "a2c":
                V, current_log_probs, entropy = self.evaluate(batch_state, batch_acts) 

                actor_loss = (-current_log_probs * A_k).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rewards_togo)
                
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            else:
                raise NotImplementedError
            
            pattern = "E {:^8d} \tActor Loss {:^8.8f} \tCritic Loss {:^8.2f}"
            print(pattern.format(episode, actor_loss, critic_loss))
            
            self.logging["Actor Loss"].append(actor_loss.item())
            self.logging["Critic Loss"].append(critic_loss.item())

        return rewards2go

    def rollout(self):
        
        # Batch data
        batch_state = []            # batch states
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rewards = []          # batch rewards
        batch_rewards_togo = []     # batch rewards-to-go
        batch_lengths = []          # episodic lengths in batch

        # Number of timesteps run so far in this batch
        t = 0

        # Sum of rewards achieved so far
        sum_rewards = []

        # default 4800
        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rewards = []
            state = self.env.reset()
            done = False

            # default 1600
            for ep_t in range(self.max_timesteps_per_episode):

                # Render state
                #if self.t_so_far > 400000:
                #    self.env.render(state)

                # Increment timesteps for this batch
                t += 1

                # Collect observation (states)
                batch_state.append(state)
                action, log_prob = self.get_action(state)

                # done is limited to 200 steps due to gym.make -> ep_t breaks after 200
                # so increment of t is in 200 steps
                state, reward, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Add summed rewards to list
            #print("Reward: ", sum(ep_rewards))
            sum_rewards.append(sum(ep_rewards))

            # Collect episodic length and rewards
            batch_lengths.append(ep_t + 1)     # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rewards)

        # Reshape data as tensors
        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_togo = self.compute_rewards_togo(batch_rewards)

        # Return batch data
        return batch_state, batch_acts, batch_log_probs, batch_rewards_togo, batch_lengths, sum_rewards

    def get_action(self, state):

        # Query actor network for mean action
        # Same as calling self.actor.forward(state)
        mean = self.actor(state)

        # Creating Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample action from distribution and get log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine.
        return action.detach().numpy(), log_prob.detach()

    def compute_rewards_togo(self, batch_rewards):

        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rewards_togo = []

        # Iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rewards in reversed(batch_rewards):

            # The discounted reward so far
            discounted_reward = 0
            for rew in reversed(ep_rewards):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rewards_togo.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rewards_togo = torch.tensor(batch_rewards_togo, dtype=torch.float)
        return batch_rewards_togo

    def evaluate(self, batch_state, batch_acts):

        # Query critic network for a value V for each state in batch_state
        V = self.critic(batch_state).squeeze()

        # Calculate log probabilites of batch actions using most recent actor network
        # Similar to get_action()
        mean = self.actor(batch_state)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()

        return V, log_probs, entropy
