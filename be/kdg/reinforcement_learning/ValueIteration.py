import math

import numpy as np

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess


class ValueIteration(LearningStrategy):
    def __init__(self, alpha, _lambda, gamma, epsilon, e_min, e_max, xi, env: TimeLimit):
        super().__init__(e_min, e_max, env)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._xi = xi
        self._v = np.zeros(self.n_states)
        self._policy = self.init_policy()

    def init_policy(self):
        arr = np.empty((0, 4))
        for i in range(self._n_states):
            arr = np.append(arr, [[0.25, 0.25, 0.25, 0.25]], 0)
        return arr

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        r_max = np.max(self.mdp.rewards)
        delta = math.inf

        while delta > self._xi * r_max * (1 - self._gamma / self._gamma):
            delta = 0
            for s in range(self.n_states):
                u = self._v[s]
                self._v[s] = max(self.value_function(s, self.mdp))
                delta = max(delta, abs(u - self._v[s]))

    def value_function(self, s, mdp: MarkovDecisionProcess):
        eu = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            eu[a] = self._policy[s, a] * self.calculate_value(s, a, mdp)
        return eu

    def calculate_value(self, s, a, mdp: MarkovDecisionProcess):
        total = 0
        for next_state in range(self.n_states):
            total += mdp.transition_model[next_state, s, a] * (mdp.rewards[s, a] + self._gamma * self._v[next_state])
        return total

    def improve(self, percept: Percept, t, policy):
        for s in range(self.n_states):
            a_stars = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                a_stars[a] = self.calculate_value(s, a, self._mdp)
            a_star = np.random.choice(np.flatnonzero(a_stars == a_stars.max()))

            for a in range(len(policy[s])):
                if a == a_star:
                    policy[s, a] = 1 - self._epsilon + (self._epsilon / self.n_actions)
                else:
                    policy[s, a] = self._epsilon / self.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)
