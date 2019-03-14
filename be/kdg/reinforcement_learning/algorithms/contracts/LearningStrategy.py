import time
from abc import ABC, abstractmethod

import numpy as np

from be.kdg.reinforcement_learning.domain import Percept, Environment
from be.kdg.reinforcement_learning.domain.MarkovDecisionProcess import MarkovDecisionProcess


class LearningStrategy(ABC):
    def __init__(self, e_min: float, e_max: float, env: Environment):
        self._e_min = e_min
        self._e_max = e_max
        self._env = env
        self._mdp = MarkovDecisionProcess(env)
        self._policy = self.init_policy()

    def init_policy(self) -> np.ndarray:
        arr = np.empty((0, self._env.n_actions))
        action_arr = np.empty(self._env.n_actions)
        p = 1 / self._env.n_actions
        for a in range(self._env.n_actions):
            action_arr[a] = p

        for i in range(self._env.n_states):
            arr = np.append(arr, [action_arr], 0)
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
