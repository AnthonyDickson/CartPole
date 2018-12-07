
import os
import argparse

import gym
        
from logger import Logger
from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Load and watch a previously trained model.')
parser.add_argument('path', type=str, help='the path of the model that is to be loaded.')
parser.add_argument('--num-episodes', type=int, default=100, help='num of episodes to playback.')

args = parser.parse_args()

logger = Logger(Logger.Verbosity.MINIMAL)
env = gym.make('CartPole-v0')
agent = CartPoleAgent.load(args.path)

for i_episode in range(args.num_episodes):
    observation = env.reset() 
    prev_observation = None
    prev_action = None

    for t in range(200):
        env.render()

        action = agent.get_action(observation)

        prev_observation = observation
        prev_action = action
        observation, reward, done, info = env.step(action)

        if done:
            msg = "Episode {:02d} finished after {:02d} timesteps".format(i_episode, t+1)
            print(msg)

            break

env.close()