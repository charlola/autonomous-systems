from src.agent.agent import Agent
import torch
import numpy as np

class PPOAgent(Agent):

    def __init__(self, params):
        Agent.__init__(self, params)

    def policy(self, state):
        return np.random.rand(1, 9)

    def update(self, state, action, reward, next_state, done):
        pass
