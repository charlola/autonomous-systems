import gym
from torch.distributions import MultivariateNormal
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

    ############################################################################

    parser.add_argument("--mode", default="train", metavar='M', type=str,
                        help='Mode to evaluate (train|test)')
    parser.add_argument("-e", "--episodes", default=100, metavar='N', type=int,
                        help='Define the number of episodes')
    parser.add_argument("-m", "--model", default="ppo.nn", metavar='S',
                        type=str, help='Define the model to be used/overwritten')
    parser.add_argument("--hidden_units", default=64, metavar='I',
                        type=int, help="Number of hidden units")
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

class Agent():
    def policy(self, state):
        return env.action_space.sample()
    
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