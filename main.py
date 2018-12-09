import argparse
from io import StringIO
import os
from pathlib import Path
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
        
from utils.annealing import Step, TReciprocal, ExponentialDecay
from utils.logger import Logger
from utils.path import get_run_path
from utils.visualisation import Dashboard
from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Train a Q-Learning agent on the CartPole problem.')
parser.add_argument('--n-episodes', type=int, default=100, help='the number of episodes to run.')
parser.add_argument('--checkpoint-rate', type=int, default=500, help='how often the logs and model should be checkpointed (in episodes). \
Set to -1 to disable checkpoints')
parser.add_argument('--render', action='store_true', help='flag to indicate the training should be rendered.')
parser.add_argument('--live-plot', action='store_true', help='flag to indicate the training data should be plotted in real-time.')
parser.add_argument('--no-plot', action='store_true', help='flag to indicate the no plot should be shown.')
parser.add_argument('--plot-update-rate', type=int, default=100, help='how often the live-plot should be updated.')
parser.add_argument('--log-verbosity', type=int, default=Logger.Verbosity.MINIMAL, choices=Logger.Verbosity.ALL, 
    help='the verbosity level of the logger.')
parser.add_argument('--model-name', type=str, default='RoleyPoley', help='the name of the model. Used as the filename when saving the model.')
parser.add_argument('--model-path', type=str, help='the path to a previous model. If this is set the designated model will be used for training.')

args = parser.parse_args()

if args.no_plot:
    args.live_plot = False

if not args.no_plot:
    dashboard = Dashboard(ema_alpha=1e-2, real_time=args.live_plot)

# Setup logger
logger = Logger(verbosity=args.log_verbosity, filename_prefix=args.model_name)
logger.log('episode_info', 'episode,timesteps')
logger.log('learning_rate', 'learning_rate')
logger.log('exploration_rate', 'exploration_rate')

# Load OpenAI Gym and agent.
env = gym.make('CartPole-v0')

model_filename = args.model_name + '.q'
checkpoint_filename_format = args.model_name + '-checkpoint-{:03d}.q'

if args.model_path:
    agent = CartPoleAgent.load(args.model_path)
    args.model_name = Path(args.model_path).name
    agent.model_path = get_run_path(prefix='data/')
else:
    agent = CartPoleAgent(env.action_space, env.observation_space, 
                            n_buckets=6, learning_rate=1, learning_rate_annealing=ExponentialDecay(k=1e-3), 
                            exploration_rate=1, exploration_rate_annealing=Step(k=2e-2, step_after=100),
                            discount_factor=0.9, input_mask=[0, 1, 1, 1])

start = time.time()

for i_episode in range(args.n_episodes):
    # per episode setup
    observation = env.reset()
    prev_observation = []
    prev_action = 0
    episode_start = time.time()

    # logging
    episode = '[Episode {}]'.format(i_episode)       
    logger.log('observations', episode)
    logger.log('rewards', episode)
    logger.log('actions', episode)

    if agent.learning_rate_annealing:
        logger.log('learning_rate', agent.learning_rate_annealing(agent.learning_rate, i_episode))
    else:
        logger.log('learning_rate', agent.learning_rate)

    if agent.exploration_rate_annealing:
        logger.log('exploration_rate', agent.exploration_rate_annealing(agent.exploration_rate, i_episode))
    else:
        logger.log('exploration_rate', agent.exploration_rate)

    # checkpointing
    if i_episode % args.checkpoint_rate  == 0:
        checkpoint = i_episode // args.checkpoint_rate 

        logger.print('Checkpoint #{}'.format(checkpoint))
        logger.print('Total elapsed time: {:02.4f}s'.format(time.time() - start))
        agent.save(checkpoint_filename_format.format(checkpoint))
        
        if not args.live_plot:
            logger.write(mode='a')
            logger.clear()
        else:
            logger.write(mode='w')

    if args.live_plot and i_episode == 1:
        dashboard.warmup(logger, agent.q_table)

    if args.live_plot and (i_episode > 0 and i_episode % args.plot_update_rate == 0):
        dashboard.draw(logger, agent.q_table)    

    cumulative_reward = 0

    for t in range(200):
        action = agent.get_action(observation, i_episode)
        logger.print('Observation:\n{}\nAction:\n{}\n'.format(observation, action), Logger.Verbosity.FULL)

        prev_observation = observation
        prev_action = action

        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        logger.print('Reward for last observation: {}'.format(cumulative_reward), Logger.Verbosity.FULL)
        agent.update(prev_observation, prev_action, cumulative_reward, observation, i_episode)

        logger.log('observations', observation)
        logger.log('rewards', reward)
        logger.log('actions', action)

        if args.render:
            env.render()

        if done:
            msg = "Episode {:02d} finished after {:02d} timesteps in {:02.4f}s".format(i_episode, t + 1, time.time() - episode_start)
            logger.log('episode_info', '{:02d}, {:02d}'.format(i_episode, t + 1))
            logger.print(msg, Logger.Verbosity.MINIMAL)

            break

env.close()
logger.write(mode='w' if args.live_plot else 'a')
agent.save(model_filename)

if args.live_plot or not args.no_plot:
    if args.live_plot:
        dashboard.draw(logger, agent.q_table)
    else:
        dashboard.warmup(logger, agent.q_table)

    dashboard.keep_on_screen()
    dashboard.close()
