import numpy
import gym

from unittest import TestCase

from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess
from be.kdg.reinforcement_learning.Percept import Percept


class tests(TestCase):
    transition_model = numpy.empty((0,4))

    def test1(self):
        markovDecisionProcess_table = MarkovDecisionProcess(self.transition_model, gym.make("FrozenLake-v0"))
        percept = Percept(0, 1, 1, 0)
        percept1 = Percept(0, 1, 1, 1)
        percept2 = Percept(0, 0, 1, 0)
        percept3 = Percept(0, 1, 2, 1)
        percept4 = Percept(0, 1, 2, 2)
        percept5 = Percept(1, 1, 1, 0)
        markovDecisionProcess_table.update(percept)
        markovDecisionProcess_table.update(percept1)
        markovDecisionProcess_table.update(percept2)
        markovDecisionProcess_table.update(percept3)
        markovDecisionProcess_table.update(percept4)
        markovDecisionProcess_table.update(percept5)
        print(markovDecisionProcess_table.rewards)
        print("===================================")
        print(markovDecisionProcess_table.n_sa)
        print("===================================")
        print(markovDecisionProcess_table.n_tsa)