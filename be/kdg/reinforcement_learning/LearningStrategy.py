import numpy as np
from abc import ABC, abstractmethod

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept


class LearningStrategy(ABC):
    def __init__(self, e_min, e_max, env: TimeLimit):
        self._e_min = e_min
        self._e_max = e_max
        self._env = env
        self._n_actions = env.action_space.n
        self._n_states = env.observation_space.n
        self._policy = self.init_policy()

    def init_policy(self):
        arr = np.empty((0, 4))
        for i in range(self.n_states):
            arr = np.append(arr, [[0.25, 0.25, 0.25, 0.25]], 0)
        return arr

    def learn(self, percept: Percept, t):
        self.evaluate(percept)
        self.improve(percept, t)

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    def improve(self, percept: Percept, t):
        pass

    @property
    def n_states(self):
        return self.n_states

    @property
    def env(self):
        return self._env

    @property
    def n_actions(self):
        return self.n_actions

    @property
    def policy(self):
        return self._policy

    @property
    def e_min(self):
        return self.e_min

    @property
    def e_max(self):
        return self.e_max
