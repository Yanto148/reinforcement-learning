import gym

from unittest import TestCase

from be.kdg.reinforcement_learning.Agent import Agent
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess


class tests(TestCase):

    def test1(self):
        env = gym.make("FrozenLake-v0")
        markovDecisionProcess_table = MarkovDecisionProcess(env)
        learning_strategy = LearningStrategy()
        agent = Agent(markovDecisionProcess_table, learning_strategy, env)
        # print(agent.policy)
        agent.learn(1000)
        print()
