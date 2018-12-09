from io import StringIO
from math import log

import matplotlib.pyplot as plt
import pandas as pd

from utils.logger import Logger
from utils.datastructures import ObservationDict

class Dashboard:
    def __init__(self, ema_alpha=0.1, real_time=True):
        self.alpha = ema_alpha
        self.fig = plt.figure(figsize=(12, 8))  
        self.was_closed = False
        
        if real_time:
            plt.ion()
            self.fig.show()

    def moving_average(self, series):
        result = [series.iloc[0]]

        for i in range(1, len(series)):
            result.append(self.alpha * series.iloc[i] + (1 - self.alpha) * result[i - 1])

        return result

    def warmup(self, log_source, q_table: ObservationDict):
        # the plot needs to be drawn and updated at least 3 times to show for some reason..
        self.draw(log_source, q_table)
        
        plt.pause(1e-8)
        plt.pause(1e-8)

    def draw(self, log_source, q_table: ObservationDict):
        """Draw the dashboard with the given data.
        
        Arguments:
            q_table: the agent's Q-value table.
            log_source: either a Logger object used in training or  a tuple containing the dataframes for *episode_info.log, *learning_rate.log, and *exploration_rate.log.
        """
        if self.was_closed:
            return

        grid_shape = (2, 5)
        try:
            if isinstance(log_source, Logger):
                episode_info, learning_rate, exploration_rate = (log_source.log_to_dataframe('episode_info'),
                                                                 log_source.log_to_dataframe('learning_rate'),
                                                                 log_source.log_to_dataframe('exploration_rate'))
            else:
                episode_info, learning_rate, exploration_rate = log_source

            ax1 = plt.subplot2grid(grid_shape, (0, 0), colspan=4, fig=self.fig)
            self.plot_file(episode_info, 'episode_info', ax1, n_cols=2, ylabel='timestep', ylim=[0, 200], moving_avg=True)

            ax2 = plt.subplot2grid(grid_shape, (1, 0), colspan=2, fig=self.fig)
            self.plot_file(learning_rate, 'learning_rate', axis=ax2, ylabel='learning rate')

            ax3 = plt.subplot2grid(grid_shape, (1, 2), colspan=2, fig=self.fig)
            self.plot_file(exploration_rate, 'exploration_rate', axis=ax3, ylabel='exploration rate')

            ax4 = plt.subplot2grid(grid_shape, (0, 4), rowspan=2, fig=self.fig)
            self.plot_qtable(q_table, ax4)

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except KeyError:
            print('dashboard was closed.')
            self.was_closed = True
        
    def plot_file(self, df, name, axis, n_cols=1, xlabel='episode', ylabel='', xlim=0, ylim=0, moving_avg=False):
        cols = tuple(df[df.columns[col]] for col in range(n_cols))        
        axis.plot(*cols)

        if moving_avg and len(cols[-1]) > 0:
            series = cols[-1]
            axis.plot(self.moving_average(series), label='moving avg. ($\\alpha=%.2f$)' % self.alpha)
            axis.legend()

        axis.set_title(name)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)

    def plot_qtable(self, q_table, axis):
        df = pd.read_csv(StringIO(q_table.to_csv()))
        
        img = axis.imshow(df.drop('observation', axis=1), cmap='hot_r')
        plt.colorbar(img, ax=axis)

        step = max(int(log(len(df))), 1)
        ticks = [i for i in range(0, len(df), step)]
        axis.set_yticks(ticks)
        axis.set_yticklabels(df['observation'][df.index % step == 0])

        axis.set_ylabel('Bucketed Observation')
        axis.set_xlabel('Aciton')
        axis.set_title('Heatmap of values in Q-Table')

    def keep_on_screen(self):
        if not self.was_closed:
            plt.ioff()
            plt.show()

    def close(self):
        plt.close(self.fig)
        plt.ioff()