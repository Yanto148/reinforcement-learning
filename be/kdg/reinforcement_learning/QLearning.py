import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.Percept import Percept
from be.kdg.reinforcement_learning.TemporalDifferenceLearning import TemporalDifferenceLearning


class QLearning(TemporalDifferenceLearning):

    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, env: TimeLimit):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, env)

    def evaluate(self, p: Percept):
        self.mdp.update(p)
        self._q[p.state, p.action] = self._q[p.state, p.action] + self._alpha * (self.mdp.rewards[p.state, p.action] + self._gamma * np.max(self._q[p.next_state]) - self._q[p.state, p.action])
