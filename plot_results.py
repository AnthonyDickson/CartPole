import argparse
import glob
from io import StringIO
from math import log
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Plot data from the log files.')
parser.add_argument('path', type=str, help='the directory that contains the log files.')
parser.add_argument('--alpha', type=float, default=0.01, help='the α term of the exponential moving average, which is used in the plot for timesteps survived.')

args = parser.parse_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError

def moving_average(series, alpha=0.9):
    result = [series.iloc[0]]

    for i in range(1, len(series)):
        result.append(alpha * series.iloc[i] + (1 - alpha) * result[i - 1])

    return result

def plot_file(path, name, n_cols=1, axis=None, xlabel='episode', ylabel='', moving_avg=False):
    if path[-1] != '/':
        path += '/'

    matching_files = list(glob.glob(path + '*' + name + '.log'))

    if len(matching_files) == 0:
        return

    filename = matching_files[0]

    df = pd.read_csv(filename, comment='[')  # ignore timestamps starting with '['

    cols = tuple(df[df.columns[col]] for col in range(n_cols))

    if axis:
        axis.plot(*cols)
    else:
        axis = plt.plot(*cols)

    if moving_avg:
        series = cols[-1]
        axis.plot(moving_average(series, alpha=args.alpha), label='moving avg. ($\\alpha=%.2f$)' % args.alpha)
        axis.legend()

    axis.set_title(name)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xlim(0)
    axis.set_ylim(0)

def plot_qtable(path, axis):
    if path[-1] != '/':
        path += '/'

    matching_files = list(glob.glob(path + '*.q'))

    if len(matching_files) == 0:
        return

    filename = matching_files[0]
    agent = CartPoleAgent.load(filename)
    df = pd.read_csv(StringIO(agent.q_table.to_csv()))

    img = axis.imshow(df.drop('observation', axis=1), cmap='hot_r')
    plt.colorbar(img, ax=axis)

    step = int(log(len(df)))
    ticks = [i for i in range(0, len(df), step)]
    axis.set_yticks(ticks)
    axis.set_yticklabels(df['observation'][df.index % step == 0])

    axis.set_ylabel('Bucketed Observation')
    axis.set_xlabel('Aciton')
    axis.set_title('Heatmap of values in Q-Table')


plt.figure(figsize=(12, 8))

grid_shape = (2, 5)

ax1 = plt.subplot2grid(grid_shape, (0, 0), colspan=4)
plot_file(args.path, 'episode_info', 2, axis=ax1, ylabel='timestep', moving_avg=True)

ax2 = plt.subplot2grid(grid_shape, (1, 0), colspan=2)
plot_file(args.path, 'exploration_rate', axis=ax2, ylabel='exploration rate')

ax3 = plt.subplot2grid(grid_shape, (1, 2), colspan=2)
plot_file(args.path, 'learning_rate', axis=ax3, ylabel='learning rate')

ax4 = plt.subplot2grid(grid_shape, (0, 4), rowspan=2)
plot_qtable(args.path, ax4)

plt.tight_layout()
plt.show()

