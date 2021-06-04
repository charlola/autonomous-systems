from src.agent.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import random
from src.ppo.config import params, hyperparams


class ReplayBuffer:
    def __init__(self, batch_size):
        self.transitions = list()
        self.batch_size = batch_size

    def save(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.size:
            self.transitions.pop(0)

    def update(self, batch_indices, minibatch, deltas):
        for index, transition, delta in zip(batch_indices, minibatch, deltas):
            self.transitions[index] = transition[:5] + (delta,)

    def sample_batch(self, minibatch_size):
        nr_episodes = len(self.transitions)
        if nr_episodes > minibatch_size:
            deltas = np.array([x[5] for x in self.transitions])
            indices = random.choices(range(len(self.transitions)), k=minibatch_size, weights=deltas)
            minibatch = [self.transitions[i] for i in indices]
            return indices, minibatch
        return range(len(self.transitions)), self.transitions

    def clear(self):
        self.transitions.clear()

    def size(self):
        return len(self.transitions)
