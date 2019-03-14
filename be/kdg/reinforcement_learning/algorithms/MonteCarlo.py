import numpy as np

from be.kdg.reinforcement_learning.domain import Environment
from be.kdg.reinforcement_learning.domain.Percept import Percept
from be.kdg.reinforcement_learning.algorithms.contracts.TemporalDifferenceLearning import TemporalDifferenceLearning


class MonteCarlo(TemporalDifferenceLearning):
    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, env: Environment):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, env)
        self._index = 0
        self._p = []
        self._n = 0

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self._p.append(percept)

        if percept.reward == 1:
            self._n = len(self._p) - self._index
            self._index = len(self._p)

        if self._n > 0:
            for i in range(len(self._p) - 1, len(self._p) - 1 - self._n, -1):
                self._q[self._p[i].state, self._p[i].action] = self._q[self._p[i].state, self._p[i].action] - self._alpha * (self._q[self._p[i].state, self._p[i].action] -
                                                (self.mdp.rewards[self._p[i].state, self._p[i].action] + self._gamma * np.max(self._q[self._p[i].next_state])))


