import gym
from torch.distributions import Normal, normal
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def collect_arguments():
    def str2bool(str):
        if str.lower() in ["false", "f", "0"]:
            return False
        return True

    parser = argparse.ArgumentParser(
        description='Define the parameter for Captain Wurmi')

    parser.add_argument("-g", "--graphics", default=False, metavar='B',
                        type=str2bool, help="Define if graphics should be shown")
    parser.add_argument("--hidden_units", default=64, metavar='I',
                        type=int, help="Number of hidden units")

    ############################################################################

    parser.add_argument("--mode", default="train", metavar='M', type=str,
                        help='Mode to evaluate (train|test)')
    parser.add_argument("-e", "--episodes", default=100, metavar='N', type=int,
                        help='Define the number of episodes')
    parser.add_argument("-m", "--model", default="ppo.nn", metavar='S',
                        type=str, help='Define the model to be used/overwritten')
    parser.add_argument("--batch_size", default=4096, metavar='I',
                        type=int, help="Size of the batch")
    parser.add_argument("--step_limit", default=1024, metavar='I',
                        type=int, help="Size of the batch")
    parser.add_argument("--gamma", default=0.99, metavar='I',
                        type=float, help="Gamma")
    parser.add_argument("--device", default="cpu", metavar='D',
                        type=torch.device, help="Choose the device to calculate")
    parser.add_argument("--ppo_episodes", default="4", metavar='I',
                        type=int, help="Number of PPO Episodes")
    parser.add_argument("--lr", default=0.003, metavar='F',
                        type=float, help="Learning Rate")
    parser.add_argument("--clip", default=0.2, metavar='F',
                        type=float, help="Clipping Value")
    parser.add_argument("--advantage", default="ADVANTAGE", metavar='A',
                        type=str, help="Choose the advantage function (REINFORCE | TEMPORAL | ADVANTAGE)")

    parser.add_argument("--action_std_init", default=1.0, metavar='F',
                        type=float, help="Set the initial action std")
    parser.add_argument("--action_std_min", default=0.1, metavar='F',
                        type=float, help="Set the minimum action std decay")
    parser.add_argument("--action_std_decay_rate", default=0.05, metavar='F',
                        type=float, help="Set the action std decay rate")
    parser.add_argument("--action_std_decay_freq", default=2.5e5, metavar='F',
                        type=int, help="Set the action std decay frequency")
    parser.add_argument("--max_grad_norm", default=0.5, metavar='F',
                        type=int, help="Maximum of gradient")
    
    return parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # get dimenstions from environment
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # get hyperparameter
        nr_hidden_units = args.hidden_units

        self.base = nn.Sequential(
            nn.Linear(obs_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(nr_hidden_units, act_dim),
            nn.Sigmoid() #TODO Which function? Tanh
        )

        self.sigma = nn.Sequential(
            nn.Linear(nr_hidden_units, act_dim),
            nn.Sigmoid() #TODO Which function? Softplus
        )

        self.value = nn.Linear(nr_hidden_units, 1)
    
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, device=device, dtype=torch.float)

        base  = self.base(state)
        mu    = self.mu(base)
        sigma = self.sigma(base)
        value = self.value(base)

        return mu, sigma, value


class Agent():
    def __init__(self):
        self.action_low  = env.action_space.low[0]
        self.action_high = env.action_space.high[0]

        self.net = Net()

    def policy(self, state):

        with torch.no_grad():
            mu, sigma, _ = self.net(state)

        dist = Normal(mu, sigma)

        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        # transform to valid action space
        action = torch.clip(action, min=0, max=1)
        action = self.action_low + action * (self.action_high - self.action_low)
        
        # check if action is valid
        assert action >= self.action_low and action <= self.action_high
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        entropy, loss, policy_loss, entropy_loss, value_loss = 0, 0, 0, 0, 0

        return entropy, loss, policy_loss, entropy_loss, value_loss

def episode(i, gamma=0.99):
    state = env.reset()
    dis_return = 0
    avg_return = None
    done = False
    t = 0
    while not done:
        if args.graphics:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        entropy, loss, policy_loss, entropy_loss, value_loss = agent.update(state, action, reward, next_state, done)

        state = next_state
        
        # calculate returns
        dis_return += (gamma**t)*reward
        if avg_return is None:
            avg_return = reward
        else:
            avg_return += (1 / (t+1)) * (reward-avg_return)

        t += 1
    string_format = "{: >3d}: R {:^12.4f} \tE {:^12.4f} \tL {:^12.4f} \tPL {:^12.4f} \tEL {:^12.4f} \tVL {:^12.4f}"
    print(string_format.format(i+1, avg_return, entropy, loss, policy_loss, entropy_loss, value_loss))
    return dis_return


def plot(returns, agent):
    pass

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args = collect_arguments()

    env = gym.make('Pendulum-v0')

    agent = Agent()

    returns = list()
    for i in range(args.episodes):
        try:
            returns.append(episode(i))
        except KeyboardInterrupt:
            break

    plot(returns, agent)