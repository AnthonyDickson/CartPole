
import os

import gym
        
from logger import Logger
from agent import CartPoleAgent

logger = Logger(Logger.Verbosity.MINIMAL)
env = gym.make('CartPole-v0')
agent = CartPoleAgent(env.action_space, env.observation_space)

for i_episode in range(20):
    observation = env.reset() 
    prev_observation = None
    prev_action = None

    episode = '[Episode {}]'.format(i_episode)       
    logger.log('observations', episode)
    logger.log('rewards', episode)
    logger.log('actions', episode)

    for t in range(200):
        env.render()

        action = agent.get_action(observation)

        prev_observation = observation
        prev_action = action
        observation, reward, done, info = env.step(action)

        agent.update(prev_observation, prev_action, reward, observation)

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