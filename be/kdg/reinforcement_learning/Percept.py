
class Percept:
    def __init__(self, state, action, next_state, reward, done):
        self._state = state
        self._action = action
        self._next_state = next_state
        self._reward = reward
        self._done = done

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def next_state(self):
        return self._next_state

    @property
    def  reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

