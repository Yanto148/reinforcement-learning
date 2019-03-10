from threading import Thread

import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.Percept import Percept


class Agent(Thread):
    def __init__(self, learning_strategy: LearningStrategy, env: TimeLimit, n_episodes) -> None:
        super().__init__()
        self._learning_strategy = learning_strategy
        self._env = env
        self._n_episodes = n_episodes
        self._n_states = env.observation_space.n
        self._policy = self.init_policy()

    def run(self) -> None:
        self.learn(self._n_episodes)

    def init_policy(self):
        arr = np.empty((0, 4))
        for i in range(self._n_states):
            arr = np.append(arr, [[0.25, 0.25, 0.25, 0.25]], 0)
        return arr

    def learn(self, n_episodes: int):
        t = 1
        for i in range(n_episodes):
            state = self._env.reset()
            episode_done = False
            while not episode_done:
                action = np.random.choice([0,1,2,3], p=self._policy[state])
                new_state, reward, done, info = self._env.step(action)
                percept = Percept(state, action, new_state, reward, done)
                self._learning_strategy.learn(percept, t, self._policy)
                state = new_state
                if done:
                    episode_done = True
                    t += 1
                    if t % 100 == 0:
                        print("===== Episode " + str(t) + " done =====")

    def visualize(self) -> np.ndarray:
        grid = self._env.unwrapped.desc.copy()
        grid = grid.astype(str)
        state = 0
        it = np.nditer(grid, op_flags=['readwrite'])
        with it:
            while not it.finished:
                if not (it[0] == 'H' or it[0] == 'G'):
                    action = np.argmax(self._policy[state])
                    # action = np.random.choice(np.flatnonzero(self._policy[state] == self._policy[state].max()))
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
    def policy(self):
        return self._policy
