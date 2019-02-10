from be.kdg.reinforcement_learning import Percept
import numpy


class MarkovDecisionProcess:
    def __init__(self, rewards, state_action_frequencies, state_action_state_frequencies, transitionModel):
        self.rewards = rewards
        self.state_action_frequencies = state_action_frequencies
        self.state_action_state_frequencies = state_action_state_frequencies
        self.transitionModel = transitionModel

    def update(self, percept: Percept):
        self.update_rewards(percept)
        self.update_state_action_frequenties(percept)
        # self.stateActionFrequencies.append()  # TODO find in array and update frequency or append
        # self.stateActionStateFrequencies.append()  # TODO same as above
        # self.transitionModel.append()  # TODO devide the 2 above

    def update_rewards(self, percept: Percept):
        if self.rewards.size == 0 or not self.reward_already_exists(percept):
            self.rewards = numpy.append(self.rewards,
                                        [[percept.previousState, percept.action, percept.currentState, percept.reward]],
                                        0)

    def update_state_action_frequenties(self, percept: Percept):
        state_action_line = self.get_state_action_line(percept)
        print(state_action_line)
        # if (state_action_line.size > 0):
        #     int(state_action_line[0,2]) + 1
        # else:
        #     self.state_action_frequencies = numpy.append(self.state_action_frequencies, [[percept.previousState, percept.action, 1]], 0)

    def reward_already_exists(self, percept: Percept) -> bool:
        check = ((self.rewards[:, 0] == str(percept.previousState)) &
                 (self.rewards[:, 1] == str(percept.action)) &
                 (self.rewards[:, 2] == str(percept.currentState)))
        return self.rewards[check].size > 0

    def get_state_action_line(self, percept: Percept):
        check = ((self.state_action_frequencies[:, 0] == str(percept.previousState)) &
                 (self.state_action_frequencies[:, 1] == str(percept.action)))
        return numpy.argwhere(self.state_action_frequencies[check])

    def state_action_state_already_exists(self, percept: Percept) -> bool:
        check = ((self.state_action_state_frequencies[:, 0] == str(percept.previousState)) &
                 (self.state_action_state_frequencies[:, 1] == str(percept.action)) &
                 (self.state_action_state_frequencies[:, 2] == str(percept.currentState)))
        return self.state_action_state_frequencies[check] > 0

