import numpy as np
from gym.wrappers import TimeLimit
from be.kdg.reinforcement_learning import Percept


class MarkovDecisionProcess:

    def __init__(self, env: TimeLimit):
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.n_sa = self.fill_3_columns()
        self.n_tsa = self.fill_4_columns()
        self.rewards = self.fill_4_columns()
        self.transition_model = self.fill_4_columns()

    def fill_3_columns(self):
        arr = np.empty((0, 3))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                arr = np.append(arr, [[i, j, 0]], 0)
        return arr

    def fill_4_columns(self):
        arr = np.empty((0, 4))
        for i in range(self.n_states):
            for j in range(self.n_actions):
                for k in range(self.n_states):
                    arr = np.append(arr, [[i, j, k, 0]], 0)
        return arr

    def update(self, percept: Percept):
        row_index = self.get_row_index(percept)
        self.update_rewards(percept)
        state_action_frequency = self.update_state_action_frequencies(percept)
        self.update_state_action_state_frequencies(percept)
        self.update_transition_model(percept, row_index, state_action_frequency)

    def update_rewards(self, percept: Percept) -> int:
        reward_line = self.get_reward_line(percept)
        row_index = np.where(np.all(self.rewards == reward_line, axis=1))[0]
        self.rewards[row_index, 3] = percept.reward
        return self.rewards[row_index, 3]

    def update_state_action_frequencies(self, percept: Percept) -> int:
        state_action_line = self.get_state_action_line(percept)
        row_index = np.where(np.all(self.n_sa == state_action_line, axis=1))[0]
        self.n_sa[row_index, 2] = int(self.n_sa[row_index, 2]) + 1
        return self.n_sa[row_index, 2]

    def update_state_action_state_frequencies(self, percept: Percept) -> int:
        state_action_state_line = self.get_state_action_state_line(percept)
        row_index = np.where(np.all(self.n_tsa == state_action_state_line, axis=1))[0]
        self.n_tsa[row_index, 3] = int(self.n_tsa[row_index, 3]) + 1
        return self.n_tsa[row_index, 3]

    def update_transition_model(self, percept: Percept, row_index: int, state_action_frequency: int):
        lower = row_index - percept.current_state
        upper = lower + self.n_states - 1
        for i in range(lower, upper):
            self.transition_model[i, 3] =  self.n_tsa[i, 3] / state_action_frequency

    def get_row_index(self, percept: Percept) -> int:
        check = ((self.n_tsa[:, 0] == percept.previous_state) &
                 (self.n_tsa[:, 1] == percept.action) &
                 (self.n_tsa[:, 2] == percept.current_state))
        line = self.n_tsa[check]
        return int(np.where(np.all(self.n_tsa == line, axis=1))[0])

    def get_reward_line(self, percept: Percept) -> bool:
        check = ((self.rewards[:, 0] == percept.previous_state) &
                 (self.rewards[:, 1] == percept.action) &
                 (self.rewards[:, 2] == percept.current_state))
        return self.rewards[check]

    def get_state_action_line(self, percept: Percept):
        check = ((self.n_sa[:, 0] == percept.previous_state) &
                 (self.n_sa[:, 1] == percept.action))
        return self.n_sa[check]

    def get_state_action_state_line(self, percept: Percept) -> bool:
        check = ((self.n_tsa[:, 0] == percept.previous_state) &
                 (self.n_tsa[:, 1] == percept.action) &
                 (self.n_tsa[:, 2] == percept.current_state))
        return self.n_tsa[check]
