import argparse
import glob
import os

import pandas as pd

from agent import CartPoleAgent
from utils.visualisation import Dashboard

parser = argparse.ArgumentParser(description='Plot data from the log files.')
parser.add_argument('path', type=str, help='the directory that contains the log files.')
parser.add_argument('--alpha', type=float, default=0.01, help='the α term of the exponential moving average, which is used in the plot for timesteps survived.')

args = parser.parse_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError

def get_df(path, name):
    if path[-1] != '/':
        path += '/'

    matching_files = list(glob.glob(path + '*' + name + '.log'))

    if len(matching_files) == 0:
        return

    filename = matching_files[0]
    df = pd.read_csv(filename, comment='[')  # ignore timestamps starting with '['

    return df

def get_agent(path):
    if path[-1] != '/':
        path += '/'

    matching_files = list(glob.glob(path + '*.q'))

    if len(matching_files) == 0:
        return

    filename = sorted(matching_files)[0]    
    agent = CartPoleAgent.load(filename)

    return agent

agent = get_agent(args.path)
dfs = (get_df(args.path, name) for name in ['episode_info', 'learning_rate', 'exploration_rate'])

db = Dashboard(real_time=False, ema_alpha=args.alpha)
db.draw(dfs, agent.q_table)
db.keep_on_screen()

