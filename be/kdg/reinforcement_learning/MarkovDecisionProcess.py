import numpy as np

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.Percept import Percept


class MarkovDecisionProcess:
    def __init__(self, env: TimeLimit):
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.n_sa = np.zeros((self.n_states, self.n_actions))
        self.n_tsa = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.rewards = np.zeros((self.n_states, self.n_actions))
        self.transition_model = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.n = 0

    def update(self, percept: Percept):
        self.n += 1
        self.update_reward(percept)
        self.update_counts(percept)
        self.update_transition_model(percept)

    def update_reward(self, p: Percept):
        self.rewards[(p.state, p.action)] = np.average([self.rewards[(p.state, p.action)], p.reward],
                                                       weights=[self.n - 1 / self.n, 1 / self.n])

    def update_counts(self, p: Percept):
        self.n_sa[p.state, p.action] += 1
        self.n_tsa[p.next_state, p.state, p.action] += 1

    def update_transition_model(self, percept: Percept):
        for t in range(self.n_states):
            p = self.n_tsa[t, percept.state, percept.action] / self.n_sa[percept.state, percept.action]
            self.transition_model[t, percept.state, percept.action] = p
