import math

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
import numpy as np

from be.kdg.reinforcement_learning.Percept import Percept


class QLearning(LearningStrategy):
    def __init__(self, mdp: MarkovDecisionProcess, a, la, ga, e, e_min, e_max, env: TimeLimit):
        super().__init__(e_min, e_max, env)
        self.mdp = mdp
        self.a = a
        self.e = e
        self.ga = ga
        self.la = la
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def evaluate(self, p: Percept):
        self.mdp.update(p)
        self.q[p.state, p.action] = self.q[p.state, p.action] + self.a * (self.mdp.rewards[p.state, p.action] + self.ga * np.max(self.q[p.next_state]) - self.q[p.state, p.action])

    def improve(self, percept: Percept, t):
        for i in range(self.env.observation_space.n):
            a_star = np.random.choice(np.flatnonzero(self.q[i] == self.q[i].max()))
            for a in range(len(self.policy[i])):
                if a == a_star:
                    self.policy[i, a] = 1 - self.e + (self.e / self.n_actions)
                else:
                    self.policy[i, a] = self.e / self.n_actions
        self.e = self.e_min + (self.e_max - self.e_min) * math.exp(-self.la * t)
