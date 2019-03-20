import numpy as np

from be.kdg.reinforcement_learning.domain import Environment
from be.kdg.reinforcement_learning.domain.Percept import Percept
from be.kdg.reinforcement_learning.algorithms.contracts.TemporalDifferenceLearning import TemporalDifferenceLearning


class QLearning(TemporalDifferenceLearning):

    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, env: Environment):
        super().__init__(_lambda, gamma, epsilon, e_min, e_max, env)
        self._alpha = alpha

    def evaluate(self, p: Percept):
        self.mdp.update(p)
        self._q[p.state, p.action] = self._q[p.state, p.action] + self._alpha * (self.mdp.rewards[p.state, p.action] + self._gamma * np.max(self._q[p.next_state]) - self._q[p.state, p.action])
