import numpy as np

from be.kdg.reinforcement_learning.domain import Percept, Environment
from be.kdg.reinforcement_learning.algorithms.contracts.TemporalDifferenceLearning import TemporalDifferenceLearning


class NStepQLearning(TemporalDifferenceLearning):
    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, n: int, env: Environment):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, env)
        self._n = n
        self._p = []

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self._p.insert(0, percept)

        if len(self._p) >= self._n:
            for i in range(0, self._n):
                self._q[self._p[i].state, self._p[i].action] = self._q[self._p[i].state, self._p[i].action] - self._alpha * (self._q[self._p[i].state, self._p[i].action] -
                                                                            (self.mdp.rewards[self._p[i].state, self._p[i].action] + self._gamma * np.max(self._q[self._p[i].next_state])))

