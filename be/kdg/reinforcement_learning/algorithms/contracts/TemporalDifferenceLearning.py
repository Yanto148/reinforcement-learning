import math
from abc import ABC, abstractmethod

import numpy as np

from be.kdg.reinforcement_learning.domain import Percept, Environment
from be.kdg.reinforcement_learning.algorithms.contracts.LearningStrategy import LearningStrategy


class TemporalDifferenceLearning(LearningStrategy, ABC):

    def __init__(self, _lambda, gamma, epsilon, e_min, e_max, env: Environment):
        super().__init__(e_min, e_max, env)
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._q = np.zeros((self._env.n_states, self._env.n_actions))

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    def improve(self, percept: Percept, t):
        for i in range(self._env.n_states):
            a_star = np.random.choice(np.flatnonzero(self._q[i] == self._q[i].max()))
            for a in range(len(self.policy[i])):
                if a == a_star:
                    self.policy[i, a] = 1 - self._epsilon + (self._epsilon / self._env.n_actions)
                else:
                    self.policy[i, a] = self._epsilon / self._env.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)
