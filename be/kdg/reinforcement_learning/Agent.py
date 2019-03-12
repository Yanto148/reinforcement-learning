from threading import Thread

import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.Percept import Percept


class Agent(Thread):
    def __init__(self, learning_strategy: LearningStrategy, env: TimeLimit, n_episodes: int):
        super().__init__()
        self._learning_strategy = learning_strategy
        self._env = env
        self._n_episodes = n_episodes
        self._n_states = env.observation_space.n

    def run(self):
        self.learn(self._n_episodes)

    def learn(self, n_episodes: int):
        t = 0
        for i in range(n_episodes):
            state = self._env.reset()
            episode_done = False
            while not episode_done:
                action = np.random.choice([0,1,2,3], p=self._learning_strategy.policy[state])
                new_state, reward, done, info = self._env.step(action)
                percept = Percept(state, action, new_state, reward, done)
                self._learning_strategy.learn(percept, t)
                state = new_state
                if done:
                    episode_done = True
                    t += 1
                    if t % 100 == 0:
                        print("===== Episode " + str(t) + " done =====")

    def visualize(self) -> np.ndarray:  # TODO move to seperrate Environment class
        grid = self._env.unwrapped.desc.copy()
        grid = grid.astype(str)
        state = 0
        it = np.nditer(grid, op_flags=['readwrite'])
        with it:
            while not it.finished:
                if not (it[0] == 'H' or it[0] == 'G'):
                    action = np.argmax(self._learning_strategy.policy[state])
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

