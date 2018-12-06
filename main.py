
import os

import gym
        
from logger import Logger

logger = Logger(Logger.Verbosity.MINIMAL)
env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()

    for t in range(100):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        logger.log('observations', observation)
        logger.log('rewards', reward)
        logger.log('actions', action)

        if done:
            msg = "Episode {:02d} finished after {:02d} timesteps".format(i_episode, t+1)
            logger.log('episode_info', msg)
            print(msg)

            break

env.close()
logger.write()