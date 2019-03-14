import math
from abc import ABC, abstractmethod

import numpy as np

from be.kdg.reinforcement_learning.domain import MarkovDecisionProcess, Percept, Environment
from be.kdg.reinforcement_learning.algorithms.contracts.LearningStrategy import LearningStrategy


class DynamicProgrammingLearning(LearningStrategy, ABC):

    def __init__(self, alpha: float, _lambda: float, gamma: float, epsilon: float, e_min: float, e_max: float, xi: float, env: Environment):
        super().__init__(e_min, e_max, env)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._xi = xi
        self._v = np.zeros(self._env.n_states)

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    def improve(self, percept: Percept, t: int):
        for s in range(self._env.n_states):
            a_stars = np.zeros(self._env.n_actions)
            for a in range(self._env.n_actions):
                a_stars[a] = self.calculate_value(s, a, self._mdp)
            a_star = np.random.choice(np.flatnonzero(a_stars == a_stars.max()))

            for a in range(len(self.policy[s])):
                if a == a_star:
                    self.policy[s, a] = 1 - self._epsilon + (self._epsilon / self._env.n_actions)
                else:
                    self.policy[s, a] = self._epsilon / self._env.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)

    def calculate_value(self, s: int, a: int, mdp: MarkovDecisionProcess) -> float:
        total = 0
        for next_state in range(self._env.n_states):
            total += mdp.transition_model[next_state, s, a] * (mdp.rewards[s, a] + self._gamma * self._v[next_state])
        return total
