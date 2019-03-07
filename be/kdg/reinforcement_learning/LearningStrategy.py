import time
from abc import ABC, abstractmethod

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess


class LearningStrategy(ABC):
    def __init__(self, e_min, e_max, env: TimeLimit):
        self._e_min = e_min
        self._e_max = e_max
        self._env = env
        self._n_actions = env.action_space.n
        self._n_states = env.observation_space.n

    def learn(self, percept: Percept, t, policy, mdp: MarkovDecisionProcess):
        self.evaluate(percept, mdp)
        self.improve(percept, t, policy)
        # To immediately show kivy, otherwise it's only shown after 200+ episodes on my machine
        # time.sleep(0.0001)

    @abstractmethod
    def evaluate(self, percept: Percept, mdp: MarkovDecisionProcess):
        pass

    def improve(self, percept: Percept, t, policy):
        pass

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def e_min(self):
        return self._e_min

    @property
    def e_max(self):
        return self._e_max
