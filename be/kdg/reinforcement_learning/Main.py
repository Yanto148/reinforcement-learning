import kivy
import gym
from gym import register

from be.kdg.reinforcement_learning.MonteCarlo import MonteCarlo
from be.kdg.reinforcement_learning.NStepQLearning import NStepQLearning
from be.kdg.reinforcement_learning.ValueIteration import ValueIteration

kivy.require('1.10.1')

from kivy.app import App
from kivy.graphics.context import Clock
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

from be.kdg.reinforcement_learning.Agent import Agent
from be.kdg.reinforcement_learning.QLearning import QLearning


class FrozenLakeWidget(GridLayout):

    def __init__(self, agent: Agent, **kwargs):
        super(FrozenLakeWidget, self).__init__(**kwargs)
        # self._disabled_count = 0
        self.agent = agent
        policy = self.agent.visualize()
        self.cols = policy.shape[1]
        for action in policy.reshape(1, -1).flatten():
            self.add_widget(
                Label(text=str(action),
                      font_size='40sp',
                      font_name='C:\Windows\Fonts\Arial.ttf')
            )

    def update(self):
        policy = self.agent.visualize()
        self.clear_widgets()
        for action in policy.reshape(1, -1).flatten():
            self.add_widget(
                Label(text=str(action),
                      font_size='40sp',
                      font_name='C:\Windows\Fonts\Arial.ttf'))


class MyApp(App):

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
        self.frozen_lake = FrozenLakeWidget(self.agent)
        Clock.schedule_interval(self.update, 1 / 30)

    def build(self):
        return self.frozen_lake

    def update(self, dt):
        self.frozen_lake.update()


if __name__ == '__main__':
    # To make environment non slippery
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )
    env = gym.make("FrozenLake-v0")
    # strategy = QLearning(0.1, 0.001, 0.9, 1, 0.05, 1, env)
    # strategy = NStepQLearning(0.1, 0.001, 0.9, 1, 0.05, 1, 5, env)
    # strategy = MonteCarlo(0.1, 0.001, 0.9, 1, 0.05, 1, env)
    strategy = ValueIteration(0.1, 0.001, 0.9, 1, 0.05, 1, 0.5, env)
    agent = Agent(strategy, env, 20000)
    # start Agent's thread
    agent.start()
    MyApp(agent).run()
