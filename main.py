import os
import argparse
import time
from pathlib import Path

import gym
        
from utils.logger import Logger
from utils.annealing import ExponentialDecay
from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Train a Q-Learning agent on the CartPole problem.')
parser.add_argument('--n-buckets', type=int, default=10, help='the number buckets to divide the observation space into.')
parser.add_argument('--n-episodes', type=int, default=100, help='the number of episodes to run.')
parser.add_argument('--checkpoint-rate', type=int, default=1000, help='how often the logs and model should be checkpointed (in episodes). \
Set to -1 to disable checkpoints')
parser.add_argument('--render', action='store_true', help='flag to indicate the training should be rendered.')
parser.add_argument('--log-verbosity', type=int, default=Logger.Verbosity.MINIMAL, choices=Logger.Verbosity.ALL, 
    help='the verbosity level of the logger.')
parser.add_argument('--model-name', type=str, default='RoleyPoley', help='the name of the model. Used as the filename when saving the model.')
parser.add_argument('--model-path', type=str, help='the path to a previous model. If this is set the designated model will be used for training.')

args = parser.parse_args()

env = gym.make('CartPole-v0')

if args.model_path:
    agent = CartPoleAgent.load(args.model_path)
    args.model_name = Path(args.model_path).name
else:
    agent = CartPoleAgent(env.action_space, env.observation_space, n_buckets=args.n_buckets, exploration_rate_annealing=ExponentialDecay())

model_filename = args.model_name + '.q'
checkpoint_filename_format = args.model_name + '-checkpoint-{:03d}.q'
logger = Logger(verbosity=args.log_verbosity, filename_prefix=args.model_name)
logger.log('episode_info', 'episode, timesteps')  # csv headings for episode info

start = time.time()

for i_episode in range(args.n_episodes):
    # per episode setup
    observation = env.reset() 
    prev_observation = None
    prev_action = None
    episode_start = time.time()

    # logging
    episode = '[Episode {}]'.format(i_episode)       
    logger.log('observations', episode)
    logger.log('rewards', episode)
    logger.log('actions', episode)

    if agent.learning_rate_annealing:
        logger.log('learning_rate', agent.learning_rate_annealing(agent.learning_rate, i_episode))

    if agent.exploration_rate_annealing:
        logger.log('exploration_rate', agent.exploration_rate_annealing(agent.exploration_rate, i_episode))

    # checkpointing
    if i_episode > 0 and i_episode % args.checkpoint_rate  == 0:
        checkpoint = i_episode // args.checkpoint_rate 

        logger.print('Checkpoint #{}'.format(checkpoint))
        logger.print('Total elapsed time: {:02.4f}s'.format(time.time() - start))
        agent.save(checkpoint_filename_format.format(checkpoint))
        logger.write(mode='a')
        logger.clear()

    for t in range(200):
        if args.render:
            env.render()

        action = agent.get_action(observation, i_episode)
        logger.print('Observation:\n{}\nAction:\n{}\n'.format(observation, action), Logger.Verbosity.FULL)
        prev_observation = observation
        prev_action = action
        observation, reward, done, info = env.step(action)
        logger.print('Reward for last observation: {}'.format(reward), Logger.Verbosity.FULL)
        agent.update(prev_observation, prev_action, reward, observation, i_episode)

        logger.log('observations', observation)
        logger.log('rewards', reward)
        logger.log('actions', action)

        if done:
            msg = "Episode {:02d} finished after {:02d} timesteps in {:02.4f}s".format(i_episode, t + 1, time.time() - episode_start)
            logger.log('episode_info', '{:02d}, {:02d}'.format(i_episode, t + 1))
            logger.print(msg, Logger.Verbosity.MINIMAL)

            break

env.close()
logger.write(mode='a')
agent.save(model_filename)