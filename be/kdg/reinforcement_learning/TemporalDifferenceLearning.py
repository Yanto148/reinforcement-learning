import math

import numpy as np
from abc import ABC, abstractmethod

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy


class TemporalDifferenceLearning(LearningStrategy, ABC):

    def __init__(self, alpha, _lambda, gamma, epsilon, e_min, e_max, env: TimeLimit):
        super().__init__(e_min, e_max, env)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._q = np.zeros((self.n_states, self.n_actions))

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    def improve(self, percept: Percept, t):
        for i in range(self.n_states):
            a_star = np.random.choice(np.flatnonzero(self._q[i] == self._q[i].max()))
            for a in range(len(self.policy[i])):
                if a == a_star:
                    self.policy[i, a] = 1 - self._epsilon + (self._epsilon / self.n_actions)
                else:
                    self.policy[i, a] = self._epsilon / self.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)
