import numpy
from unittest import TestCase

from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess
from be.kdg.reinforcement_learning.Percept import Percept


class tests(TestCase):
    rewards = numpy.empty((0,4))
    state_action_frequenties = numpy.empty((0, 3))
    state_action_state_frequenties = numpy.empty((0,4))
    transition_model = numpy.empty((0,4))

    def test1(self):
        percept = Percept(0, "right", 1, 0)
        markovDecisionProcess_table = MarkovDecisionProcess(self.rewards, self.state_action_frequenties, self.state_action_state_frequenties, self.transition_model)
        percept1 = Percept(0, "right", 1, 1)
        percept2 = Percept(0, "up", 1, 0)
        percept3 = Percept(0, "right", 2, 0)
        percept4 = Percept(1, "right", 1, 0)
        markovDecisionProcess_table.update(percept)
        markovDecisionProcess_table.update(percept1)
        markovDecisionProcess_table.update(percept2)
        markovDecisionProcess_table.update(percept3)
        markovDecisionProcess_table.update(percept4)
        # p = markovDecisionProcess_table.state_action_already_exists(percept4)
        print(markovDecisionProcess_table.state_action_frequencies)
        # print(p)