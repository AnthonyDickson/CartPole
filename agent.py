import os
import unittest

import numpy as np
import pickle
from gym import make
from gym.spaces import Discrete, Box

from utils import Utils

class CartPoleAgent:
    """A Q-Learning agent for the cart-pole problem.
    
    The observation space for the cart pole problem is continuous so the agent buckets (discretises) the observation data.
    """
    def __init__(self, action_space: Discrete, observation_space: Box, n_buckets: int=100, learning_rate=0.1, discount_rate=0.9):
        """Setup the agent.

        Arguments:
            action_space: The action space of the cart-pole enviroment.
            observation_space: the observation space of the cart-pole enviroment.
            n_buckets: how many buckets to separate the observation space into (since it is continuous).
            learning_rate: how much a recent experience affects future actions.
        """
        self.actions = np.arange(0, action_space.n)
        self.n_buckets = n_buckets
        self.low = observation_space.low
        self.high = observation_space.high
        # self.q_table = np.zeros((n_buckets ** observation_space.shape[0], action_space.n), dtype=observation_space.dtype)
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.model_path = ''

    def get_action(self, observation):
        """Get the best action based on the current observation.
        
        Arguments:
            observation: a set of observation values from the environment.

        Returns: the optimal action (integer) based on the current q_table
        """
        return np.argmax([self.q(observation, action) for action in self.actions])

    def update(self, prev_observation, prev_action, reward, observation):
        """Update the Q-value for the previous observation and action.
        
        Arguments:
            prev_observation: the set of observation values from the previous step.
            prev_action: the action taken last step.
            reward: the reward from taking the previous action.
            observation: the set of observation values for the next step.
        """
        old_Q = self.q(prev_observation, prev_action)
        action = self.get_action(observation)
        next_Q = self.q(observation, action)       

        a = self.learning_rate
        g = self.discount_rate
        self.q_table_cell(prev_observation)[prev_action] = (1 - a) * old_Q  + a * reward + g * next_Q

    def q_table_cell(self, observation):
        """Find the cell in the q_table for the given observation.
        
        Arguments:
            observation: a set of observation values from the environment.
        
        Missing cells are lazily created.
        """
        a, b, c, d = self.get_buckets(observation)

        cell = self.q_table

        for i, key in enumerate([a, b, c, d]):
            try:
                cell = cell[key]
            except KeyError:
                cell[key] = {} if i < 3 else np.zeros(len(self.actions))
                cell = cell[key]

        return cell  
    
    def q(self, observation, action):
        """Get the Q-value for the given observation and action.
        
        Arguments:
            observation: a set of observation values from the environment.
            action: the action of the Q-value that is to be retrieved.

        Returns: the Q-value for the given observation and action.

        """
        return self.q_table_cell(observation)[action]    

    def get_buckets(self, observation):
        """Get the buckets for each value in the observation.
        
        Arguments:
            observation: a set of observation values from the environment.

        Returns: the observation as bucketed values.
        """
        bucketed = []

        for i, o in enumerate(observation):
            bucketed.append(self.get_bucket(o, i))

        return np.array(bucketed)

    def get_bucket(self, observation_value, axis=0):
        """Convert an observation value into a discrete value via bucketing.
        
        Arguments:
            observation_value: the value to discretise.
            axis: the nth dimension in the observation space where each dimension is one of the observation variables.

        Returns: the observation as a bucketed value.
        """
        step_size = abs(self.low[axis]) / self.n_buckets + abs(self.high[axis]) / self.n_buckets

        for bucket in range(self.n_buckets):
            if self.low[axis] + bucket * step_size <= observation_value < self.low[axis] + (bucket + 1) * step_size:
                return bucket

        if bucket == self.high[axis]:
            return self.n_buckets
        
        return -1

    def save(self, filename='RoleyPoley.q'):
        """Save the agent's current state to file.

        Arguments:
            filename: the name of the file to be saved.

        Returns: the path of to where the file was saved.
        """
        path = Utils.get_run_path(prefix='models/')
        os.makedirs(path, exist_ok=True)
        self.model_path = path + filename

        with open(path + filename, 'wb') as f:
            f.write(pickle.dumps(self))

        return self.model_path

    @staticmethod
    def load(fullpath):
        """Load a saved model.

        Arguments:
            fullpath: the path including the filename of the saved model.

        Returns: the saved model at the designated path.
        """
        with open(fullpath, 'rb') as f:
            return pickle.load(f)    
