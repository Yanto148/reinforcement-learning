import math

import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.DynamicProgrammingLearning import DynamicProgrammingLearning
from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess


class ValueIteration(DynamicProgrammingLearning):

    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, xi: float, env: TimeLimit):
        super().__init__(alpha, _lambda, gamma, epsilon, e_min, e_max, xi, env)

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        # r_max = np.max(self.mdp.rewards)
        r_max = 1       # TODO
        delta = math.inf

        while delta > self._xi * r_max * ((1 - self._gamma) / self._gamma):
            delta = 0
            for s in range(self.n_states):
                u = self._v[s]
                self._v[s] = max(self.value_function(s, self.mdp))
                delta = max(delta, abs(u - self._v[s]))

    def value_function(self, s, mdp: MarkovDecisionProcess) -> np.ndarray:
        eu = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            eu[a] = self.policy[s, a] * self.calculate_value(s, a, mdp)
        return eu
