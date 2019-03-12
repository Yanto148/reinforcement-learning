import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.TemporalDifferenceLearning import TemporalDifferenceLearning


class NStepQLearning(TemporalDifferenceLearning):
    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, n: int, env: TimeLimit):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, env)
        self._n = n
        self._p = []

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self._p.append(percept)

        if len(self._p) >= self._n:
            for i in range(len(self._p) - self._n, len(self._p)):
                self._q[self._p[i].state, self._p[i].action] = self._q[self._p[i].state, self._p[i].action] - self._alpha * (self._q[self._p[i].state, self._p[i].action] -
                                                               (self.mdp.rewards[self._p[i].state, self._p[i].action] + self._gamma * np.max(self._q[self._p[i].next_state])))

