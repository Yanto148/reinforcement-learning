import math

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
import numpy as np

from be.kdg.reinforcement_learning.Percept import Percept


class QLearning(LearningStrategy):
    def __init__(self, alpha, _lambda, gamma, epsilon, e_min, e_max, env: TimeLimit):
        super().__init__(e_min, e_max, env)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = _lambda
        self._q = np.zeros((self.n_states, self.n_actions))

    def evaluate(self, p: Percept, mdp: MarkovDecisionProcess):
        mdp.update(p)
        self._q[p.state, p.action] = self._q[p.state, p.action] + self._alpha * (mdp.rewards[p.state, p.action] + self._gamma * np.max(self._q[p.next_state]) - self._q[p.state, p.action])

    def improve(self, percept: Percept, t, policy):
        for i in range(self.n_states):
            a_star = np.random.choice(np.flatnonzero(self._q[i] == self._q[i].max()))
            for a in range(len(policy[i])):
                if a == a_star:
                    policy[i, a] = 1 - self._epsilon + (self._epsilon / self.n_actions)
                else:
                    policy[i, a] = self._epsilon / self.n_actions
        self._epsilon = self.e_min + (self.e_max - self.e_min) * math.exp(-self._lambda * t)
