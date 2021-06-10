import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy
from src.agent.agent import Agent

class Net(nn.Module):
    def __init__(self, state_dim, nr_hidden_units, action_dim):
        super(Net, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(state_dim, nr_hidden_units),
            nn.ReLU(), 
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )

        self.mean_net = nn.Sequential(
            nn.Linear(nr_hidden_units, action_dim),
            #nn.Tanh(),
        )

        self.sigma_net = nn.Sequential(
            nn.Linear(nr_hidden_units, action_dim),
            nn.Softplus(),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, nr_hidden_units),
            nn.ReLU(), 
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1)
        )
    
    def forward(self, states):
        action_base = self.base(states)
        mean  = self.mean_net(action_base)
        sigma = self.sigma_net(action_base)

        value = self.critic_net(states)

        return mean, sigma, value

class ActorCritic():
    def __init__(self, params, device="cpu"):
        self.device = device
        
        self.action_low  = params["env"].min_action
        self.action_high = params["env"].max_action
        action_dim = params["env"].action_space.shape[0]
        state_dim  = params["env"].observation_space.shape[0]
        nr_hidden_units = params['nr_hidden_units'] 

        self.net = Net(state_dim, nr_hidden_units, action_dim)

        self.optimizer  = torch.optim.Adam(self.net.parameters(), lr=params["alpha"])

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)

        # collect distribution parameter
        with torch.no_grad():
            mean, sigma, _ = self.net(states)

        # define distribution   
        dist = Normal(mean, sigma)

        # sample action from distribution 
        action = dist.sample()#.cpu().numpy().flatten()

        # collect probabilities
        action_logprob = dist.log_prob(action)

        # transform action values to action space
        action = torch.clamp(action, min=self.action_low, max=self.action_high)
        
        return action.detach(), action_logprob.detach()

    def critic(self, states):
        with torch.no_grad():
            _, _, value = self.net(states)
            
        return value.detach().cpu().numpy().flatten()

    def collect(self, state, action):
        # collect parameter
        mean, sigma, value = self.net(state)
        
        # define distribution
        dist = Normal(mean, sigma)

        # collect probabilities
        logprob = dist.log_prob(action)

        # collect entropy
        entropy  = dist.entropy()
        
        return value, logprob, entropy

    def copy_weights(self, other):
        self.net.load_state_dict(deepcopy(other.net.state_dict()))
        self.net.eval()

    def save(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        weight = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(weight)
        self.net.eval()