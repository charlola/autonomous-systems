import gym
from torch.distributions import Normal, normal
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math


def collect_arguments():
    def str2bool(str):
        if str.lower() in ["false", "f", "0"]:
            return False
        return True

    parser = argparse.ArgumentParser(
        description='Define the parameter for Captain Wurmi')
    
    # helper
    parser.add_argument("-e", "--episodes", default=100, metavar='N', type=int, help='Define the number of episodes')
    parser.add_argument("-g", "--graphics", default=False, metavar='B', type=str2bool, help="Define if graphics should be shown")
    
    # net
    parser.add_argument("--hidden_units", default=64, metavar='I', type=int, help="Number of hidden units")

    # hyperparameter
    parser.add_argument("--gamma", default=0.99,  metavar='I', type=float, help="Gamma")
    parser.add_argument("--lr",    default=0.0005, metavar='F', type=float, help="Learning Rate")
    parser.add_argument("--noise", default=0.001, metavar='F', type=float, help="Noise Factor")
    parser.add_argument("--value", default=0.5,   metavar='F', type=float, help="Value Factor")

    ############################################################################

    parser.add_argument("--mode", default="train", metavar='M', type=str,
                        help='Mode to evaluate (train|test)')
    parser.add_argument("--model", default="ppo.nn", metavar='S',
                        type=str, help='Define the model to be used/overwritten')
    parser.add_argument("--batch_size", default=4096, metavar='I',
                        type=int, help="Size of the batch")
    parser.add_argument("--step_limit", default=1024, metavar='I',
                        type=int, help="Size of the batch")
    parser.add_argument("--ppo_episodes", default="4", metavar='I',
                        type=int, help="Number of PPO Episodes")
    parser.add_argument("--clip", default=0.2, metavar='F',
                        type=float, help="Clipping Value")
    parser.add_argument("--advantage", default="ADVANTAGE", metavar='A',
                        type=str, help="Choose the advantage function (REINFORCE | TEMPORAL | ADVANTAGE)")

    
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

class Buffer():
    def __init__(self):
        self.actions = list()
        self.values = list()
        self.logprobs = list()
        self.states = list()
        self.next_states = list()
        self.rewards = list()
        self.mus = list()
        self.sigmas = list()
        self.entropies = list()
        self.dones = list()

    def add_policy(self, action, logprob, entropy, mu, sigma, value):
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.entropies.append(entropy)
        self.mus.append(mu)
        self.sigmas.append(sigma)
        self.values.append(value)

    def add_update(self, state, next_state, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.actions.clear()
        self.values.clear()
        self.logprobs.clear()
        self.states.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.mus.clear()
        self.sigmas.clear()
        self.entropies.clear()
        self.dones.clear()



class Agent():
    def __init__(self):
        self.action_low  = env.action_space.low[0]
        self.action_high = env.action_space.high[0]

        self.net = Net()
        self.buffer = Buffer()
        self.logger = dict()
        self.logger["total_reward"] = list()
        self.logger["loss"]  = list()
        self.logger["actor_loss"]  = list()
        self.logger["critic_loss"] = list()
        self.logger["entropy"] = list()
                
        self.mse = nn.MSELoss()

        self.optimizer = Adam(self.net.parameters(), lr=args.lr)

    def policy(self, state):

        with torch.no_grad():
            mu, sigma, value = self.net(state)

        dist = Normal(mu, sigma)

        action   = dist.sample()
        logprob = dist.log_prob(action)
        entropy  = dist.entropy()

        # transform to valid action space
        action = torch.clip(action, min=0, max=1)
        action = self.action_low + action * (self.action_high - self.action_low)
        
        # check if action is valid
        assert action >= self.action_low and action <= self.action_high
        
        self.buffer.add_policy(action, logprob, entropy, mu, sigma, value)

        return action
    
    def update(self, state, action, reward, next_state, done):
        self.buffer.add_update(state, next_state, reward, done)

        if done:
            self.learn()
            self.buffer.clear()


    def learn(self):
        actions     = torch.tensor(self.buffer.actions,     device=device, dtype=torch.float)
        logprobs    = torch.tensor(self.buffer.logprobs,    device=device, dtype=torch.float)#.unsqueeze(1)
        states      = torch.tensor(self.buffer.states,      device=device, dtype=torch.float)
        next_states = torch.tensor(self.buffer.next_states, device=device, dtype=torch.float)
        mus         = torch.tensor(self.buffer.mus,         device=device, dtype=torch.float)
        sigmas      = torch.tensor(self.buffer.sigmas,      device=device, dtype=torch.float)
        values      = torch.tensor(self.buffer.values,      device=device, dtype=torch.float)
        entropies   = torch.tensor(self.buffer.entropies,   device=device, dtype=torch.float)
        rewards     = torch.tensor(self.buffer.rewards,     device=device, dtype=torch.float)#.unsqueeze(1)
        dones       = np.array(self.buffer.dones)
        not_dones   = (np.array(self.buffer.dones) + 1) % 2
        
        if False:
            print("actions    ", actions.shape)
            print("logprobs   ", logprobs.shape)
            print("states     ", states.shape)
            print("next_states", next_states.shape)
            print("mus        ", mus.shape)
            print("sigmas     ", sigmas.shape)
            print("entropies  ", entropies.shape)
            print("rewards    ", rewards.shape)
            exit()
        
        # calculate discounted return
        dis_returns = list()
        R = 0
        for r in reversed(rewards):
            R = r + args.gamma * R
            dis_returns.insert(0, R)

        """
        LEARN STUFF
        """
        
        self.optimizer.zero_grad()
        mus, sigmas, values = self.net(states)
        next_mus, next_sigmas, next_values = self.net(next_states)

        #advantages = rewards + args.gamma * next_values.detach() - values.detach()
        advantages = rewards - values.detach()
        
        p1 = -((actions - mus) ** 2) / (2 * sigmas.clamp(min=1e-3)) # TODO video didnt use sigma**2
        p2 = -torch.log(torch.sqrt(2 * math.pi * sigmas)) # TODO video didnt use sigma**2
        calc_logprobs = (p1 + p2)
        #print(logprobs)
        #print(calc_logprobs)

        policy_loss = -(logprobs * advantages).mean() # TODO evtl calc_logprobs instead of logprobs

        value_loss  = args.value * self.mse(values.squeeze(-1), rewards) # TODO evtl discounted returns

        entropy_calc = -(torch.log(2*math.pi*sigmas) + 1) / 2

        entropy_loss = args.noise * -entropies.mean()
        
        loss = policy_loss + value_loss + entropy_loss

        # create gradient and apply to net
        loss.backward()
        self.optimizer.step()

        entropy = entropies.mean()

        self.logger["total_reward"].append(rewards.sum())
        self.logger["loss"].append(loss)
        self.logger["actor_loss"].append(policy_loss)
        self.logger["critic_loss"].append(value_loss)
        self.logger["entropy"].append(entropy)

        # Logging
        string_format = "E {:^12.4f} \tL {:^12.4f} \tPL {:^12.4f} \tEL {:^12.4f} \tVL {:^12.4f} \t"
        print(string_format.format(entropy, loss, policy_loss, entropy_loss, value_loss), end="")

def episode(i, gamma=0.99):
    state = env.reset()
    dis_return = 0
    avg_return = None
    done = False
    t = 0
    string_format = "{: >3d}: "
    print(string_format.format(i+1), end="")
    while not done:
        if args.graphics:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)

        state = next_state
        
        # calculate returns
        dis_return += (gamma**t)*reward
        if avg_return is None:
            avg_return = reward
        else:
            avg_return += (1 / (t+1)) * (reward-avg_return)

        t += 1
    string_format = "R {:^12.4f}"
    print(string_format.format(avg_return))

    return dis_return
    
def show_episode():
    state = env.reset()
    done = False
    while not done:
        env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        state = next_state
    


def plot(returns, smoothing=0.9, use_average=True, start_avg=10):
    if len(returns) < start_avg: 
        print("To Short for plot")
        return

    columns = 2
    fig, axs = plt.subplots(int((columns-1+1+len(agent.logger))/columns), 2, figsize=(10, 6), constrained_layout=True)
    
    x = range(len(returns)-start_avg+1)
    y = [sum(returns[:start_avg]) / start_avg]

    for t, r in enumerate(returns[start_avg:]):
        if use_average:
            temp = y[-1] + (1/(t+1)) * (r-y[-1])
        else:
            temp = y[-1] * smoothing + r * (1-smoothing)
        y.append(temp)
    
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title("discounted return")

    for i, (name, values) in enumerate(agent.logger.items(), start=1):
        xi = i % columns
        yi = int(i / columns)

        x = range(len(values)-start_avg+1)
        y = [sum(values[:start_avg]) / start_avg]

        for t, r in enumerate(values[start_avg:]):
            if use_average:
                temp = y[-1] + (1/(t+1)) * (r-y[-1])
            else:
                temp = y[-1] * smoothing + r * (1-smoothing)
            y.append(temp)
        
        axs[yi, xi].plot(x, y)
        axs[yi, xi].set_title(name)
    plt.show()

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

    plot(returns)

    for i in range(1):
        show_episode()
