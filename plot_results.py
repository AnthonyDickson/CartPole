import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Plot data from the log files.')
parser.add_argument('path', type=str, help='the directory that contains the log files.')

args = parser.parse_args()

if not os.path.isdir(args.path):
    raise FileNotFoundError

def plot_file(path, name, n_cols=1, axis=None, xlabel='', ylabel=''):
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

    axis.set_title(name)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xlim(0)
    axis.set_ylim(0)


plt.figure(figsize=(8, 6))

ax1 = plt.subplot(212)
plot_file(args.path, 'episode_info', 2, axis=ax1, xlabel='episode', ylabel='timestep')

ax2 = plt.subplot(221)
plot_file(args.path, 'exploration_rate', axis=ax2, xlabel='episode', ylabel='exploration rate')

ax3 = plt.subplot(222)
plot_file(args.path, 'learning_rate', axis=ax3, xlabel='episode', ylabel='learning rate')

plt.show()

