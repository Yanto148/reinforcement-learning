from be.kdg.reinforcement_learning import Percept
import numpy


class MarkovDecisionProcess:
    def __init__(self, rewards, stateActionFrequencies, stateActionStateFrequencies, transitionModel):
        self.rewards = rewards
        self.stateActionFrequencies = stateActionFrequencies
        self.stateActionStateFrequencies = stateActionStateFrequencies
        self.transitionModel = transitionModel

    def update(self, percept: Percept):
        self.rewards.append(percept.previousState, percept.action,
                            percept.currentState)  # TODO only if reward does not exist yet
        self.stateActionFrequencies.append()  # TODO find in array and update frequency or append
        self.stateActionStateFrequencies.append()  # TODO same as above
        self.transitionModel.append()  # TODO devide the 2 above
