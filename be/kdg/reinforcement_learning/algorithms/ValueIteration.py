import math

import numpy as np

from be.kdg.reinforcement_learning.domain import Percept, Environment
from be.kdg.reinforcement_learning.algorithms.contracts.DynamicProgrammingLearning import DynamicProgrammingLearning
from be.kdg.reinforcement_learning.domain.MarkovDecisionProcess import MarkovDecisionProcess


class ValueIteration(DynamicProgrammingLearning):

    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, xi: float, env: Environment):
        super().__init__(_lambda, gamma, epsilon, e_min, e_max, xi, env)
        self._alpha = alpha

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        r_max = np.max(self.mdp.rewards)
        delta = math.inf

        while delta > self._xi * r_max * ((1 - self._gamma) / self._gamma):
            delta = 0
            for s in range(self._env.n_states):
                u = self._v[s]
                self._v[s] = max(self.value_function(s, self.mdp))
                delta = max(delta, abs(u - self._v[s]))

    def value_function(self, s, mdp: MarkovDecisionProcess) -> np.ndarray:
        eu = np.zeros(self._env.n_actions)
        for a in range(self._env.n_actions):
            eu[a] = self.policy[s, a] * self.calculate_value(s, a, mdp)
        return eu
