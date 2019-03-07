from unittest import TestCase

import gym

from be.kdg.reinforcement_learning.Agent import Agent
from be.kdg.reinforcement_learning.QLearning import QLearning


# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3


class tests(TestCase):
    def test1(self):
        # To make environment non slippery
        # register(
        #     id='FrozenLakeNotSlippery-v0',
        #     entry_point='gym.envs.toy_text:FrozenLakeEnv',
        #     kwargs={'map_name': '4x4', 'is_slippery': False},
        #     max_episode_steps=100,
        #     reward_threshold=0.78,  # optimum = .8196
        # )
        env = gym.make("FrozenLake-v0")
        env.reset()
        learning_strategy = QLearning(0.1, 0.001, 0.9, 1, 0.05, 1, env)
        agent = Agent(learning_strategy, env, 5000)
        agent.learn(5000)
        print(agent.policy)

