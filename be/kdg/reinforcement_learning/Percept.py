class Percept:
    def __init__(self, previousState: int, action: str, currentState: int, reward: int):
        self.previousState = previousState
        self.action = action
        self.currentState = currentState
        self.reward = reward

