
import argparse
import os
from time import time, sleep

import gym

from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Load and watch a previously trained model.')
parser.add_argument('path', type=str, help='the path of the model that is to be loaded.')
parser.add_argument('--n-episodes', type=int, default=20, help='num of episodes to playback.')
parser.add_argument('--fps', type=int, default=100, help='frame rate for rendering. Set to -1 to render as fast as possible.')

args = parser.parse_args()
frame_delay = 1.0 / args.fps

env = gym.make('CartPole-v0')
agent = CartPoleAgent.load(args.path)

for i_episode in range(args.n_episodes):
    observation = env.reset() 
    prev_observation = None
    prev_action = None

    for t in range(200):
        start = time()
        env.render()

        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode {:02d} finished after {:02d} timesteps".format(i_episode, t+1))

            break

        if frame_delay > 0:
            delta_time = time() - start
            sleep(max(frame_delay - delta_time, 0))


env.close()