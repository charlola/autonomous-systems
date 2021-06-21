import torch
import numpy as np
from torch.distributions import MultivariateNormal

from network import Net

class PPO:
    def __init__(self, args):
        self.args = args

        # Environment information
        self.env = args.env

        # Initialize actor and critic networks
        self.actor  = Net(args.state_dim, args.hidden_units, args.act_dim)
        self.critic = Net(args.state_dim, args.hidden_units, 1)

        # Create our variable for the matrix
        # Chose 0.5 for standarddeviation
        # Create covariance matrix
        self.cov_var = torch.full(size=(args.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Initialize optimizer
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.logging = {name:[] for name in ["Total Reward", "Average Reward", "Actor Loss", "Critic Loss"]}

    def learn(self, episodes):

        # Timesteps simulated so far
        self.t_so_far = 0

        episode = 0
        batch = 0

        # Collect rewards
        rewards2go = []

        while episode < episodes:
            try:
                # Perform rollout to get batches
                batch_state, batch_acts, batch_log_probs, batch_rewards_togo, batch_lengths, sum_rewards = self.rollout()

                # Sum up rewards and add to list
                rewards2go += sum_rewards

                # Calculate how many timesteps were collected this batch
                self.t_so_far += np.sum(batch_lengths)

                episode += len(sum_rewards)
                batch   += 1
                
                # Evaluate state and actions
                V, _, entropy = self.evaluate(batch_state, batch_acts)

                # Calculate Advantage
                A_k = batch_rewards_togo - V.detach()

                # Normalize Advantages (Trick: makes PPO more stable)
                # Subtracting 1e-10, so there will be no possibility of dividing by 0
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
                
                
                if self.args.algorithm == "ppo":
                    # default at 5 updates per iteration
                    for _ in range(self.args.k):

                        # Evaluate state and actions to calculate V_phi and pi_theta(a_t | s_t)
                        V, current_log_probs, entropy = self.evaluate(batch_state, batch_acts)

                        # Calculate ratios
                        ratios = torch.exp(current_log_probs - batch_log_probs)

                        # Calculate surrogate losses
                        surr1 = ratios * A_k
                        surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

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
        
                elif self.args.algorithm == "a2c":
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
                
                R = np.mean(sum_rewards)
                
                pattern = "Batch {: >4d} Episode {: >8d} \tRewards {: >12.2f} \tActor Loss {: >12.6f} \tCritic Loss {: >12.2f}"
                print(pattern.format(batch, episode, R, actor_loss, critic_loss))
                
                self.logging["Total Reward"].extend(sum_rewards)
                self.logging["Average Reward"].append(R)
                self.logging["Actor Loss"].append(actor_loss.item())
                self.logging["Critic Loss"].append(critic_loss.item())

                    

            except KeyboardInterrupt:
                break

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
        while t < self.args.batch_size:

            # Rewards this episode
            ep_rewards = []
            state = self.env.reset()
            done = False

            # default 1600
            for ep_t in range(self.args.max_step):

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
                discounted_reward = rew + discounted_reward * self.args.gamma
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
