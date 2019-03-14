from threading import Thread

import numpy as np

from be.kdg.reinforcement_learning.domain import Environment
from be.kdg.reinforcement_learning.algorithms.contracts.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.domain.Percept import Percept


class Agent(Thread):
    def __init__(self, learning_strategy: LearningStrategy, env: Environment, n_episodes: int):
        super().__init__()
        self._learning_strategy = learning_strategy
        self._env = env
        self._n_episodes = n_episodes
        self._reward_list = []

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
                    self._reward_list.append(percept.reward)
                    episode_done = True
                    t += 1
                    if t % 100 == 0:
                        average = np.mean(self._reward_list)
                        print("Episode: {} - Reward average: {}".format(t, average, self._learning_strategy))
                        self._reward_list.clear()

    @property
    def learning_strategy(self):
        return self._learning_strategy
