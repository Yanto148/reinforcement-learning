import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.Percept import Percept
from be.kdg.reinforcement_learning.TemporalDifferenceLearning import TemporalDifferenceLearning


class MonteCarlo(TemporalDifferenceLearning):
    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, env: TimeLimit):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, env)
        self._index = 0
        self._p = []

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self._p.append(percept)

        if percept.reward == 1:
            for i in range(self._index, len(self._p)):
                self._q[self._p[i].state, self._p[i].action] = self._q[self._p[i].state, self._p[i].action] - self._alpha * (self._q[self._p[i].state, self._p[i].action] -
                                                (self.mdp.rewards[self._p[i].state, self._p[i].action] + self._gamma * np.max(self._q[self._p[i].next_state])))
            self._index = len(self._p)

