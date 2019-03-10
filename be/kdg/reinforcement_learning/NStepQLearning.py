import math

import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy


class NStepQLearning(LearningStrategy):
    def __init__(self, alpha, _lambda, gamma, epsilon, e_min, e_max, n, env: TimeLimit):
        super().__init__(e_min, e_max, env)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._q = np.zeros((self.n_states, self.n_actions))
        self._n = n
        self._p = []

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self._p.append(percept)

        if len(self._p) >= self._n:
            for i in range(len(self._p) - self._n, len(self._p)):
                self._q[self._p[i].state, self._p[i].action] = self._q[self._p[i].state, self._p[i].action] - self._alpha * (self._q[self._p[i].state, self._p[i].action] -
                                                               (self.mdp.rewards[self._p[i].state, self._p[i].action] + self._gamma * np.max(self._q[self._p[i].next_state])))

    def improve(self, percept: Percept, t, policy):
        for i in range(self.n_states):
            a_star = np.random.choice(np.flatnonzero(self._q[i] == self._q[i].max()))
            for a in range(len(policy[i])):
                if a == a_star:
                    policy[i, a] = 1 - self._epsilon + (self._epsilon / self.n_actions)
                else:
                    policy[i, a] = self._epsilon / self.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)

