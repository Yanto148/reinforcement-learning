
class Percept:
    def __init__(self, state: int, action: int, next_state: int, reward: float, done: bool):
        self._state = state
        self._action = action
        self._next_state = next_state
        self._reward = reward
        self._done = done

    @property
    def state(self) -> int:
        return self._state

    @property
    def action(self) -> int:
        return self._action

    @property
    def next_state(self) -> int:
        return self._next_state

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

