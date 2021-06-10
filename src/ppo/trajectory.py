class Trajectory:
    def __init__(self):
        self.actions = list()
        self.logprobs = list()
        self.states = list()
        self.next_states = list()
        self.rewards = list()

    def size(self):
        return len(self.rewards)

    def add_policy(self, action, logprob):
        self.actions.append(action)
        self.logprobs.append(logprob)

    def add_update(self, state, next_state, reward):
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)

    def clear(self):
        self.actions.clear()
        self.logprobs.clear()
        self.states.clear()
        self.next_states.clear()
        self.rewards.clear()