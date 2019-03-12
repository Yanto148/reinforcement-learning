import numpy as np

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.Percept import Percept


class MarkovDecisionProcess:
    def __init__(self, env: TimeLimit):
        self._n_actions = env.action_space.n
        self._n_states = env.observation_space.n
        self._n_sa = np.zeros((self._n_states, self._n_actions))
        self._n_tsa = np.zeros((self._n_states, self._n_states, self._n_actions))
        self._rewards = np.zeros((self._n_states, self._n_actions))
        self._transition_model = np.zeros((self._n_states, self._n_states, self._n_actions))
        self._n = 0

    def update(self, percept: Percept):
        self._n += 1
        self.update_reward(percept)
        self.update_counts(percept)
        self.update_transition_model(percept)

    def update_reward(self, p: Percept):
        self._rewards[(p.state, p.action)] = np.average([self._rewards[(p.state, p.action)], p.reward], weights=[self._n - 1 / self._n, 1 / self._n])

    def update_counts(self, p: Percept):
        self._n_sa[p.state, p.action] += 1
        self._n_tsa[p.next_state, p.state, p.action] += 1

    def update_transition_model(self, percept: Percept):
        for t in range(self._n_states):
            p = self._n_tsa[t, percept.state, percept.action] / self._n_sa[percept.state, percept.action]
            self._transition_model[t, percept.state, percept.action] = p

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards

    @property
    def transition_model(self) -> np.ndarray:
        return self._transition_model
