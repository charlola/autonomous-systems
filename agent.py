from abc import ABC

import torch
import numpy as np

class Agent(ABC):
    def __init__(self, args, model):
        self.args = args 

        self.mse = torch.nn.MSELoss()
        self.model = model
    
    def get_action(self, state):
        #return action, log_prob
        raise NotImplementedError

    def policy(self, state):
        return self.get_action(state)[0]
    
    def update(self, state, action, reward, next_state, done):
        pass

    def rollout(self):
        
        # Batch data
        states    = []
        actions   = []
        log_probs = []
        rewards   = []

        # Number of timesteps run so far in this batch
        t = 0

        # Sum of rewards achieved so far
        sum_rewards = []

        # default 4800
        while t < self.args.batch_size:

            # Rewards this episode
            ep_states    = []
            ep_actions   = []
            ep_log_probs = []
            ep_rewards = []
            state = self.args.env.reset()
            done = False

            ep_t = 0
            while True:

                # Increment timesteps for this batch
                t += 1
                ep_t += 1

                # Get action 
                action, log_prob = self.model.get_action(state)

                # done is limited to 200 steps due to gym.make -> ep_t breaks after 200
                # so increment of t is in 200 steps
                next_state, reward, done, _ = self.args.env.step(action)

                # Collect observation (state), reward, action, and log prob
                ep_states.append(state)
                ep_rewards.append(reward)
                ep_actions.append(action)
                ep_log_probs.append(log_prob)

                state = next_state

                if done or (self.args.max_step > 0 and ep_t >= self.args.max_step):
                    break
            
            padding = self.args.max_step - ep_t

            if padding > 0 and self.args.max_step > 0:
                print("Insert zero padding x", padding)

                # apply zero padding
                def create(sample):
                    if type(sample) == int: return 0
                    else: return np.zeros_like(sample)
                
                for _ in range(padding):
                    ep_states.insert(0,     create(state))
                    ep_actions.insert(0,    create(action))
                    ep_log_probs.insert(0,  0)
                    ep_rewards.insert(0,    0)

            # Add summed rewards to list
            sum_rewards.append(sum(ep_rewards))

            # Collect episodic rewards
            rewards.append(ep_rewards)
            states.extend(ep_states)
            actions.extend(ep_actions)
            log_probs.extend(ep_log_probs)

        # calculate discounted return
        discounted_return = self.compute_rewards_togo(rewards)

        # Reshape data as tensors
        states            = torch.tensor(states,            dtype=torch.float)
        actions           = torch.tensor(actions,           dtype=torch.float)
        log_probs         = torch.tensor(log_probs,         dtype=torch.float)
        rewards           = torch.tensor(rewards,           dtype=torch.float)
        discounted_return = torch.tensor(discounted_return, dtype=torch.float)

        # Return batch data
        return states, actions, log_probs, rewards, sum_rewards, discounted_return

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

        return batch_rewards_togo