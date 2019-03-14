import numpy as np

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.algorithms.contracts.LearningStrategy import LearningStrategy


class Environment:

    def __init__(self, env: TimeLimit):
        self._env = env
        self._n_actions = self._env.action_space.n
        self._n_states = self._env.observation_space.n

    def visualize(self, learning_strategy: LearningStrategy) -> np.ndarray:
        grid = self._env.unwrapped.desc.copy()
        grid = grid.astype(str)
        state = 0
        it = np.nditer(grid, op_flags=['readwrite'])
        with it:
            while not it.finished:
                if not (it[0] == 'H' or it[0] == 'G'):
                    action = np.argmax(learning_strategy.policy[state])
                    if action == 0:
                        action = u"\u2190"
                    elif action == 1:
                        action = u"\u2193"
                    elif action == 2:
                        action = u"\u2192"
                    else:
                        action = u"\u2191"
                    it[0] = action
                state += 1
                it.iternext()
        return grid

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_states(self):
        return self._n_states
