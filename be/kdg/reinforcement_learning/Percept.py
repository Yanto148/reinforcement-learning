class Percept:
    def __init__(self, previousState: int, action: str, currentState: int, reward: int):
        self.previousState = previousState
        self.action = action
        self.currentState = currentState
        self.reward = reward

    @property
    def previousState(self):
        return self.previousState

    @property
    def action(self):
        return self.action

    @property
    def currentState(self):
        return self.currentState

    @property
    def reward(self):
        return self.reward

