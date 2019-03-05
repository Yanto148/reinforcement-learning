import numpy as np
from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.Percept import Percept


class Agent:
    def __init__(self, learning_strategy: LearningStrategy, env: TimeLimit) -> None:
        self._learning_strategy = learning_strategy
        self._env = env

    def learn(self, n_episodes: int):
        t = 1
        for i in range(n_episodes):
            state = self.env.reset()
            episode_done = False
            while not episode_done:
                action = np.random.choice([0,1,2,3], p=self.learning_strategy.policy[state])
                new_state, reward, done, info = self.env.step(action)
                percept = Percept(state, action, new_state, reward, done)
                self.learning_strategy.learn(percept, t)
                state = new_state
                if (done):
                    episode_done = True
                    t += 1

    @property
    def learning_strategy(self):
        return self._learning_strategy

    @property
    def env(self):
        return self.env
