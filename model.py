from abc import ABC
from abc import abstractmethod

from network import Net, ActorNet
import torch
import os
import pickle
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np

class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        self.actor = None 
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
    
    def check(self):
        assert self.actor is not None 
        assert self.critic is not None
        assert self.actor_optimizer is not None
        assert self.critic_optimizer is not None
        
    def save(self, checkpoint_path, logger):
        self.actor.save(checkpoint_path + "_actor.nn")
        self.critic.save(checkpoint_path + "_critic.nn")
        with open(checkpoint_path + "_data.log", "wb") as f:
            pickle.dump(logger, f)

    def load(self, checkpoint_path, logger):
        self.actor.load(checkpoint_path + "_actor.nn")
        self.critic.load(checkpoint_path + "_critic.nn")

        if os.path.isfile(checkpoint_path + "_data.log"):
            with open(checkpoint_path + "_data.log", "rb") as f:
                old_logger = pickle.load(f)
            logger.update(old_logger)
    
    def optimize(self, actor_loss, critic_loss):
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    @abstractmethod
    def get_base(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_mean(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_covmat(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_value(self, state):
        raise NotImplementedError

    def distribution(self, states):
        base   = self.get_base(states)
        mean   = self.get_mean(base)
        covmat = self.get_covmat(base)

        # Creating Multivariate Normal Distribution
        return MultivariateNormal(mean, covmat)

    def get_action(self, states):
        # convert state to tensor if it's a numpy array
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float)

        # Creating Multivariate Normal Distribution
        dist = self.distribution(states)

        # Sample action from distribution and get log prob
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine.
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, states, actions):
        # convert state to tensor if it's a numpy array
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float)

        # Creating Multivariate Normal Distribution
        dist      = self.distribution(states)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy().mean()
        V         = self.get_value(states)
        return V, log_probs, entropy


class Model(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)
        
        # Initialize actor and critic networks
        self.actor  = Net(args.device, args.state_dim, args.hidden_units, args.act_dim, args.activation)
        self.critic = Net(args.device, args.state_dim, args.hidden_units, 1, args.activation)

        # Initialize optimizer
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=args.actor_lr,  device=self.args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, device=self.args.device)

        # Create our variable for the matrix
        # Chose 0.5 for standarddeviation
        # Create covariance matrix
        self.cov_var = torch.full(size=(args.act_dim,), fill_value=0.5, device=self.args.device)
        self.cov_mat = torch.diag(self.cov_var, device=self.args.device)
        
        self.check()

    def get_base(self, states):
        return states

    def get_mean(self, states):
        # Query actor network for mean action
        # Same as calling self.actor.forward(state)
        return self.actor.net(states)

    def get_covmat(self, states):
        return self.cov_mat

    def get_value(self, states):
        return self.critic.net(states).squeeze()

class AdvancedModel(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)

        # Initialize actor and critic networks
        self.actor  = ActorNet(args.device, args.state_dim, args.hidden_units, args.act_dim, args.activation)
        self.critic = Net(args.device, args.state_dim, args.hidden_units, 1, args.activation)

        # Initialize optimizer
        self.actor_optimizer  = torch.optim.Adam(list(self.actor.net.parameters()) + list(self.actor.mean.parameters()) + list(self.actor.std.parameters()), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.net.parameters(), lr=args.critic_lr)
        
        self.check()

    def get_base(self, state):
        return self.actor.net(state)
        
    def get_mean(self, base):
        return torch.tanh(self.actor.mean(base))

    def get_covmat(self, base):
        var = F.softplus(self.actor.std(base))
        cov_mat = torch.diag_embed(var)
        return cov_mat

    def get_value(self, states):
        return self.critic.net(states).squeeze()