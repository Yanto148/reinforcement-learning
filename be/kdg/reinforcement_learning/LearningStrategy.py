from abc import ABC

from be.kdg.reinforcement_learning import Percept


class LearningStrategy(ABC):
    def learn(self, percept: Percept):
        self.evaluate(percept)
        self.improve()

    def evaluate(self, percep: Percept):
        pass

    def improve(self):
        pass
