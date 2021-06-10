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
        #return np.random.rand(9,1)
        action, logprob = self.net_backup.act([state])

        # collect action and its probability
        self.trajectory.add_policy(action, logprob)

        return action

    def update(self, state, action, reward, next_state, done):
        # collect states and rewards
        self.trajectory.add_update(state, next_state, reward)

        # train model if necessary (done or reached step size)
        if done or self.trajectory.size() == self.step_size:
            self.train()

    def train(self):
        # wrap as tensors
        states      = torch.tensor(self.trajectory.states, device=self.device, dtype=torch.float)
        next_states = torch.tensor(self.trajectory.next_states, device=self.device, dtype=torch.float)
        actions     = torch.tensor(self.trajectory.actions, device=self.device, dtype=torch.float)
        logprobs    = torch.tensor(self.trajectory.logprobs, device=self.device, dtype=torch.float)
        
        # calculate advantage temporal difference actor-critic 
        rewards    = self.normalize_rewards(self.trajectory.rewards)

        # TODO evtl andere loop sachen 
        for _ in range(self.k):

            advantages = self.calculate_advantage(rewards, states, next_states)
            
            values, new_logprobs, entropy = self.net.collect(states, actions)
            
            actor_loss  = self.get_actor_loss(new_logprobs, logprobs, advantages)
            critic_loss = self.get_critic_loss(values, rewards)
            noise       = self.get_noise(entropy)

            # TODO split actor and critic - maybe this causes the -10 confusion
            loss = actor_loss - noise + critic_loss 

            # take gradient step
            self.net.optimize(torch.mean(loss))

        # Copy new weights into old policy
        self.net_backup.copy_weights(self.net)

        # Clear buffer
        self.trajectory.clear()

    def sample(self):
        arr = np.arange(self.trajectory.size())
        np.random.shuffle(arr)
        return arr[:self.batch_size]

    def save(self, path):
        self.net.save(path)

    def calculate_advantage(self, rewards, states, next_states):
        # REINFORCE
        #return rewards

        Q_values      = self.net_backup.critic(states)
        advantages    = (rewards - Q_values).detach()
        return advantages

        # temporal difference 
        Q_values      = self.net_backup.critic(states)
        next_Q_values = self.net_backup.critic(next_states)
        advantages    = (rewards + self.gamma * next_Q_values - Q_values).detach()
        return advantages
    
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
        
        return norm_returns.unsqueeze(1)

    def get_actor_loss(self, new_logprob, old_logprob, advantage):
        # calculate difference between old and now policy probability
        ratio = torch.exp(new_logprob-old_logprob)
        
        # TODO !!! min(p1, p2) wenn advantage positiv oder negativ ist dann unterschiedliche wahl ?!

        p1 = ratio * advantage
        p2 = ratio.clip(1-self.epsilon, 1+self.epsilon) * advantage
        actor_loss = -torch.min(p1, p2)
        return actor_loss

    def get_critic_loss(self, Q_value, reward):
        critic_loss = self.mse(Q_value, reward)
        return critic_loss * self.critic_discount

    def get_noise(self, entropy):
        # TODO !!! eventuell wie epsilon decay noise_factor reduzieren
        return entropy * self.noise_factor