class Percept:
    def __init__(self, previous_state: int, action: int, current_state: int, reward: int):
        self.previous_state = previous_state
        self.action = action
        self.current_state = current_state
        self.reward = reward
