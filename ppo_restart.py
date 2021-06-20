#############
# ARGUMENTS #
#############

from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import argparse


def collect_arguments():
    def str2bool(str):
        if str.lower() in ["false", "f", "0"]:
            return False
        return True

    parser = argparse.ArgumentParser(
        description='Define the parameter for Captain Wurmi')
    parser.add_argument("--mode", default="train", metavar='M', type=str,
                        help='Mode to evaluate (train|test)')
    parser.add_argument("-e", "--episodes", default=100, metavar='N', type=int,
                        help='Define the number of episodes')
    parser.add_argument("-m", "--model", default="ppo.nn", metavar='S',
                        type=str, help='Define the model to be used/overwritten')
    parser.add_argument("-g", "--graphics", default=True, metavar='B',
                        type=str2bool, help="Define if graphics should be shown")
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

################
# NEURONAL NET #
################
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNet, self).__init__()

        nr_hidden_units = args.hidden_units

        self.net = nn.Sequential(
            nn.Linear(input_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, output_dim),
            nn.Tanh()
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        return self.net(state)
        
class CriticNet(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(CriticNet, self).__init__()

        nr_hidden_units = args.hidden_units

        self.net = nn.Sequential(
            nn.Linear(input_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, output_dim)
        )

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        return self.net(state)

##########
# BUFFER #
##########

class Buffer:
    def __init__(self):
        self.counter = 0
        self.actions = list()
        self.logprobs = list()
        self.states = list()
        self.next_states = list()
        self.rewards = list()
        self.episode_rewards = list()
        self.dones = list()

    def size(self):
        return len(self.rewards)

    def add_policy(self, action, logprob):
        self.counter += 1
        self.actions.append(action)
        self.logprobs.append(logprob)

    def add_update(self, state, reward, next_state, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.episode_rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.actions.clear()
        self.logprobs.clear()
        self.states.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.episode_rewards.clear()
        self.dones.clear()

################
# ACTOR CRITIC #
################

class ActorCritic:
    def __init__(self):
        # Extract environment information
        self.env = env
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Create actor and critic
        self.actor_net  = ActorNet(self.obs_dim, self.act_dim)
        self.critic_net = CriticNet(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim  = Adam(self.actor_net.parameters(),  lr=args.lr)
        self.critic_optim = Adam(self.critic_net.parameters(), lr=args.lr)
        
        # Initialize the covariance matrix used to query the actor for actions
        self.set_action_std(args.action_std_init)
        #self.cov_mat = torch.diag(self.action_var)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.action_var = torch.full((self.act_dim,), new_action_std * new_action_std)
    
    def decay_action_std(self):
        if args.action_std_decay_rate == 0: return

        self.action_std = self.action_std - args.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= args.action_std_min):
            self.action_std = args.action_std_min
            print("setting actor output action_std to action_std_min : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        
    def save(self, checkpoint_path):
        torch.save(self.actor_net.state_dict(),  "actor.nn")
        torch.save(self.critic_net.state_dict(), "critic.nn")
   
    def load(self, checkpoint_path):
        weight = torch.load("actor", map_location=lambda storage, loc: storage)
        self.actor_net.load_state_dict(weight)
        self.actor_net.eval()
        
        weight = torch.load("critic", map_location=lambda storage, loc: storage)
        self.critic_net.load_state_dict(weight)
        self.critic_net.eval()

    def copy(self, other):
        self.actor_net.load_state_dict(other.actor_net.state_dict())
        self.actor_net.eval()
        
        self.critic_net.load_state_dict(other.critic_net.state_dict())
        self.critic_net.eval()

    def act(self, state):

        # Query the actor network for a mean action
        mean = self.actor_net(state)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        cov_mat = torch.diag(self.action_var)#.unsqueeze(dim=0)
        dist = MultivariateNormal(mean, cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for the sampled action
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()

    def critic(self, state):
        V = self.critic_net(state)

        return V.detach()
        
    def evaluate(self, actions, states):

        # Query the actor network for a mean action
        mean = self.actor_net(states)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        action_var = self.action_var.expand_as(mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(mean, cov_mat)

        logprobs = dist.log_prob(actions)
        
        V = self.critic_net(states)

        return V, logprobs

    def optimize(self, actor_loss, critic_loss):
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), args.max_grad_norm)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), args.max_grad_norm)
        self.critic_optim.step()

#########
# AGENT #
#########

class Agent:

    def __init__(self):
        self.actor_critic        = ActorCritic()
        self.actor_critic_backup = ActorCritic()

        # Copy weights to backup
        self.actor_critic_backup.copy(self.actor_critic)

        # Create buffer
        self.buffer = Buffer()

        # Create loss function
        self.mse = nn.MSELoss()

        self.logger = dict()
        self.logger["std"] = list()
        self.logger["loss"] = list()
        self.logger["actor_loss"] = list()
        self.logger["critic_loss"] = list()

    def policy(self, state):
        action, log_prob = self.actor_critic_backup.act(state) 

        # Collect action and its probability
        self.buffer.add_policy(action, log_prob)
        
        # Return the sampled action and the log probability of that action
        return action

    def update(self, state, action, reward, next_state, done):
        self.buffer.add_update(state, reward, next_state, done)

        if self.buffer.counter % args.action_std_decay_freq == 0:
            self.actor_critic.decay_action_std()

        if done or len(self.buffer.episode_rewards) >= args.step_limit:
            # Discount rewards
            R_discounted = []
            R_prev = 0
            for R in reversed(self.buffer.episode_rewards):
                R_prev = R + args.gamma * R_prev
                R_discounted.insert(0, R_prev)
            
            self.buffer.rewards.extend(R_discounted)
            self.buffer.episode_rewards.clear()

            next_state = env.reset()

        if self.buffer.size() >= args.batch_size:
            done = True
            self.learn()
        else:
            done = False

        return done, next_state

    def learn(self):
        actions     = torch.tensor(self.buffer.actions,     dtype=torch.float)
        logprobs    = torch.tensor(self.buffer.logprobs,    dtype=torch.float).unsqueeze(1)
        states      = torch.tensor(self.buffer.states,      dtype=torch.float)
        next_states = torch.tensor(self.buffer.next_states, dtype=torch.float)
        rewards     = torch.tensor(self.buffer.rewards,     dtype=torch.float).unsqueeze(1)
        dones       = np.array(self.buffer.dones)
        not_dones   = (np.array(self.buffer.dones) + 1) % 2
                
        # Calculate Advantage
        V = self.actor_critic.critic(states).detach()

        if args.advantage == "REINFORCE":
            A = V
        elif args.advantage == "TEMPORAL":
            V_next = self.actor_critic.critic(next_states).detach()
            A = torch.zeros_like(rewards)
            A[not_dones] = rewards + args.gamma * V_next - V
            A[dones]     = rewards                       - V
        else:
            A = rewards - V

        # Normalize Advantage
        #A = (A - A.mean()) / (A.std() + 1e-10)

        # Update the network for k epochs
        for k in range(args.ppo_episodes):
            
            V, new_logprobs = self.actor_critic_backup.evaluate(actions, states)

            # Calculate ration between new and old probabilities
            ratios = torch.exp(new_logprobs - logprobs)

            # Calculate surrogate losses
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - args.clip, 1 + args.clip) * A

            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = self.mse(V, rewards)

            # optimize actor critic
            self.actor_critic.optimize(actor_loss, critic_loss)
        
        print("Actor Loss at", self.buffer.counter, "is", actor_loss.item())
        self.logger["actor_loss"].append(actor_loss.item())
        
        # Clear trajectories from buffer
        self.buffer.clear()
        
        # Copy weights to backup
        self.actor_critic_backup.copy(self.actor_critic)

########
# MAIN #
########

def episode(env, agent, nr_episode):
    state = env.reset()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        if args.graphics:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        if args.mode == "train":
            done, state = agent.update(state, action, reward, next_state, done)
        
        discounted_return += (0.99**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return

########
# PLOT #
########

def plot(returns, agent, smoothing=0.9, use_average=True):

    x = range(len(returns))
    y = returns[:1]

    for t, r in enumerate(returns[1:]):
        if use_average:
            temp = y[-1] + (1/(t+1)) * (r-y[-1])
        else:
            temp = y[-1] * smoothing + r * (1-smoothing)
        y.append(temp)
    
    plt.plot(x, y)
    plt.title("Progress")
    plt.xlabel("episode")
    plt.ylabel("undiscounted return")
    plt.show()

    for name in ["actor_loss"]:#, "critic_loss"]:
        x = range(len(agent.logger[name]))
        y = agent.logger[name]
        plt.clf()
        plt.plot(x, y)
        plt.title("Progress")
        plt.xlabel("timestamp")
        plt.ylabel(name)
        plt.show()

    # close environment
    env.close()


if __name__ == "__main__":
    args = collect_arguments()

    env = gym.make('Pendulum-v0')

    agent = Agent()

    returns = list()
    for i in range(args.episodes):
        try:
            returns.append(episode(env, agent, i))
        except KeyboardInterrupt:
            break

    plot(returns, agent)
