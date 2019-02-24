import numpy as np

from gym.wrappers import TimeLimit

from be.kdg.reinforcement_learning.MarkovDecisionProcess import MarkovDecisionProcess
from be.kdg.reinforcement_learning.LearningStrategy import LearningStrategy
from be.kdg.reinforcement_learning.Percept import Percept



class Agent:
    def __init__(self, mdp: MarkovDecisionProcess, learning_strategy: LearningStrategy, env: TimeLimit) -> None:
        self.mdp = mdp
        self.learning_strtategy = learning_strategy
        self.n_states = env.observation_space.n
        self.policy = self.init_policy()
        self.env = env

    def init_policy(self):
        arr = np.empty((0, 4))
        for i in range(self.n_states):
            arr = np.append(arr, [[0.25, 0.25, 0.25, 0.25]], 0)
        return arr

    def learn(self, n_episodes: int):

        for i in range(n_episodes):
            state = self.env.reset()
            episode_done = False
            while not episode_done:
                action = np.random.choice([0,1,2,3], p=[0.25,0.25,0.25,0.25])
                new_state, reward, done, info = self.env.step(action)
                percept = Percept(state, action, new_state, reward)
                self.mdp.update(percept)
                self.env.render()
                state = new_state
                if (done):
                    episode_done = True
                    print("====Epidode " + str(i) + " done====")
