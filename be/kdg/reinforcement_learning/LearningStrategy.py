import numpy as np
import time
from abc import ABC, abstractmethod

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning import Percept
from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess


class LearningStrategy(ABC):
    def __init__(self, e_min: float, e_max: float, env: TimeLimit):
        self._e_min = e_min
        self._e_max = e_max
        self._env = env
        self._n_actions = env.action_space.n
        self._n_states = env.observation_space.n
        self._mdp = MarkovDecisionProcess(env)
        self._policy = self.init_policy()

    def init_policy(self) -> np.ndarray:
        arr = np.empty((0, 4))          # TODO remove magic numbers
        for i in range(self._n_states):
            arr = np.append(arr, [[0.25, 0.25, 0.25, 0.25]], 0) # TODO remove magic numbers
        return arr

    def learn(self, percept: Percept, t: int):
        self.evaluate(percept)
        self.improve(percept, t)
        # To immediately show kivy, otherwise it's only shown after 200+ episodes on my machine. This will slow things down though.
        time.sleep(0.0001)

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    @abstractmethod
    def improve(self, percept: Percept, t: int):
        pass

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def e_min(self) -> float:
        return self._e_min

    @property
    def e_max(self) -> float:
        return self._e_max

    @property
    def mdp(self) -> MarkovDecisionProcess:
        return self._mdp

    @property
    def policy(self) -> np.ndarray:
        return self._policy
