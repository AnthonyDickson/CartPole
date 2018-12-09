import argparse
from io import StringIO
from math import log
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import CartPoleAgent

parser = argparse.ArgumentParser(description='Load and watch a previously trained model.')
parser.add_argument('path', type=str, default='', help='the path of the model that is to be loaded.')
args = parser.parse_args()


def plot_qtable(agent):
        df = pd.read_csv(StringIO(agent.q_table.to_csv()))

        plt.figure(figsize=(8, 6))

        ax1 = plt.subplot(121)
        ax1.axis('tight')
        ax1.axis('off')

        text = []

        n_rows, n_cols = df.values.shape

        for row in range(n_rows):
            text.append([])

            for col in range(n_cols):
                text[row].append('{:02.4f}'.format(df.values[row, col]).rstrip('0').rstrip('.'))

        ax1.table(cellText=text, colLabels=df.columns, loc='center')
        
        ax2 = plt.subplot(122)
        img = ax2.imshow(df.drop('observation', axis=1), cmap='hot_r')
        plt.colorbar(img, ax=ax2)

        step = int(log(len(df)))
        ticks = [i for i in range(0, len(df), step)]
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(df['observation'][df.index % step == 0])

        ax2.set_ylabel('Bucketed Observation')
        ax2.set_xlabel('Aciton')
        ax2.set_title('Heatmap of values in Q-Table')

        plt.tight_layout()
        plt.show()

agent = CartPoleAgent.load(args.path)
plot_qtable(agent)