import os
import torch
import torch.nn as nn
import numpy as np
from src.agent.agent import Agent
from src.ppo.actorcritic import ActorCritic
from src.ppo.trajectory import Trajectory

class PPOAgent(Agent):

    def __init__(self, params):
        Agent.__init__(self, params)

        # choose device (gpu if availiable)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        # collect hyperparameter
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon"]
        self.step_size = params["step_size"]
        self.batch_size = params["batch_size"]
        self.k = params["k"]
        self.critic_discount = params["critic_discount"]
        self.noise_factor = params["noise_factor"]
        self.eps = np.finfo(np.float32).eps.item()
        
        # create actor critic and backup for stabilization
        self.net        = ActorCritic(params=params, device=self.device)
        self.net_backup = ActorCritic(params=params, device=self.device)
        
        # load model from file
        if "model" in params and os.path.isfile(params["model"]):
            self.net.load(params["model"])

        # match backup with original net
        self.net_backup.copy_weights(self.net)

        # create loss function object
        self.mse = nn.MSELoss()

        # create buffer to store steps data
        self.trajectory = Trajectory()

    def policy(self, state):
        """Choose an action based on the actor net"""

        states = torch.tensor(state, device=self.device, dtype=torch.float)

        #return np.random.rand(9,1)
        action, logprob = self.net_backup.act(states)

        # collect action and its probability
        self.trajectory.add_policy(action, logprob)
        
        return action

    def update(self, state, action, reward, next_state, done):
        state      = torch.tensor(state,      dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        
        # collect states and rewards
        self.trajectory.add_update(state, next_state, reward)

        # train model if necessary (done or reached step size)
        if done or self.trajectory.size() == self.step_size:
            self.train()

    def train(self):
        # wrap as tensors
        states      = torch.squeeze(torch.stack(self.trajectory.states,      dim=0)).detach().to(self.device)
        next_states = torch.squeeze(torch.stack(self.trajectory.next_states, dim=0)).detach().to(self.device)
        actions     = torch.squeeze(torch.stack(self.trajectory.actions,     dim=0)).detach().to(self.device)
        logprobs    = torch.squeeze(torch.stack(self.trajectory.logprobs,    dim=0)).detach().to(self.device)
        
        rewards    = self.normalize_rewards(self.trajectory.rewards)
        advantages = self.calculate_advantage(rewards, states, next_states)
        rewards = rewards.unsqueeze(1)

        batches = self.create_batches()

        for k in range(self.k):

            # Calculate losses            
            index = batches[k]
            index = np.arange(self.trajectory.size())
            
            values, new_logprobs, entropy = self.net.collect(states[index], actions[index])
            
            actor_loss  = self.get_actor_loss(new_logprobs, logprobs[index], advantages[index])
            critic_loss = self.get_critic_loss(values, rewards[index])
            noise       = self.get_noise(entropy)
            
            loss = actor_loss + critic_loss # - noise

            # take gradient step
            self.net.optimize(loss.mean())

        # Copy new weights into old policy
        self.net_backup.copy_weights(self.net)

        # Clear buffer
        self.trajectory.clear()

    def create_batches(self):
        arr = np.arange(self.trajectory.size())
        np.random.shuffle(arr)

        batches = list()
        for i in range(self.k):
            batch = arr[self.batch_size*i:self.batch_size*(i+1)]
            batches.append(np.array(batch))
        return batches

    def save(self, path):
        self.net.save(path)

    def calculate_advantage(self, rewards, states, next_states):
        # REINFORCE
        #return rewards

        # advantage
        Q_values   = self.net_backup.critic(states)
        advantages = rewards - Q_values
        return advantages.detach()

        # temporal difference advantage
        Q_values      = self.net_backup.critic(states)
        next_Q_values = self.net_backup.critic(next_states)

        advantages    = (rewards + self.gamma * next_Q_values - Q_values).detach()
        
        return advantages.unsqueeze(1)
    
    def normalize_rewards(self, rewards):
        """Normalize rewards"""

        # Calculate discounted returns
        disc_returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            disc_returns.insert(0, R)

        # Normalizing the discounted rewards
        disc_returns = torch.tensor(disc_returns, device=self.device, dtype=torch.float32)
        norm_returns = (disc_returns - disc_returns.mean()) / (disc_returns.std() + self.eps)
        
        return norm_returns

    def get_actor_loss(self, new_logprob, old_logprob, advantage):
        # calculate difference between old and now policy probability
        ratio = torch.exp(new_logprob-old_logprob)
            
        # TODO !!! min(p1, p2) wenn advantage positiv oder negativ ist dann unterschiedliche wahl ?!
        p1 = ratio * advantage
        p2 = ratio.clip(1-self.epsilon, 1+self.epsilon) * advantage
        actor_loss = -torch.min(p1, p2)#.mean()
        return actor_loss

    def get_critic_loss(self, Q_value, reward):
        critic_loss = self.mse(Q_value, reward)
        return critic_loss * self.critic_discount

    def get_noise(self, entropy):
        # TODO !!! eventuell wie epsilon decay noise_factor reduzieren
        return entropy * self.noise_factor