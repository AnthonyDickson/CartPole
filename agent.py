import os
import unittest
from datetime import datetime

import numpy as np
import pickle
from gym import make
from gym.spaces import Discrete, Box

from utils import Utils, ObservationDict, MultiBucketer 

class CartPoleAgent:
    """A Q-Learning agent for the cart-pole problem.
    
    The observation space for the cart pole problem is continuous so the agent buckets (discretises) the observation data.
    """
    def __init__(self, action_space: Discrete, observation_space: Box, n_buckets: int=100, learning_rate=0.1, discount_rate=0.9,
        initial_q_value = 0):
        """Setup the agent.

        Arguments:
            action_space: The action space of the cart-pole enviroment.
            observation_space: the observation space of the cart-pole enviroment.
            n_buckets: how many buckets to separate the observation space into (since it is continuous).
            learning_rate: how much a recent experience affects future actions.
        """
        self.bucketer = MultiBucketer(observation_space.low, observation_space.high, n_buckets)
        self.actions = np.arange(0, action_space.n)
        self.action_counts = ObservationDict(0, action_space.n, self.bucketer)
        self.q_table = ObservationDict(initial_q_value, action_space.n, self.bucketer)
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.model_path = ''

    def get_action(self, observation):
        """Get the best action based on the current observation.
        
        Arguments:
            observation: a set of observation values from the environment.

        Returns: the optimal action (integer) based on the current q_table
        """
        # UCB-1 first chooses any actions that have yet to be chosen at least once.
        for action in self.actions:
            if self.action_counts[observation][action] < 1:
                self.action_counts[observation][action] = 1

                return action

        q_values = [self.q_table[observation][action] + self.bonus(observation, action) for action in self.actions]
        action = np.argmax(q_values)
        self.action_counts[observation][action] += 1

        return action

    def update(self, prev_observation, prev_action, reward, observation):
        """Update the Q-value for the previous observation and action.
        
        Arguments:
            prev_observation: the set of observation values from the previous step.
            prev_action: the action taken last step.
            reward: the reward from taking the previous action.
            observation: the set of observation values for the next step.
        """
        old_Q = self.q_table[prev_observation][prev_action]
        action = self.get_action(observation)
        next_Q = self.q_table[observation][action]

        a = self.learning_rate
        g = self.discount_rate
        self.q_table[prev_observation][prev_action] = (1 - a) * old_Q  + a * reward + g * next_Q  

    def bonus(self, observation, action, C=1.0):
        """Calculate the exploration bonus for the observation-action pair.

        Based on the UCB-1 exploration method where the bonus is defined as:

            bonus(st, ai) = 100 × C × √(2 × log N(st) / N(st, ai)

        where st is the state (or observation) at timestep t,
            ai is the an action from the action space,
            C is a constant that increases the amount of exploration,
            N(st) is the number of times an action was chosen for the chosen state,
            N(st, ai) is the number of times an action was chosen under the chosen state.

        """
        N_st = self.observation_count(observation)
        N_st_ai = self.action_counts[observation][action]

        return 100.0 * C * np.sqrt(2 * np.log(N_st) / N_st_ai)

    def observation_count(self, observation):
        """Compute the sum of action_count(observation, action) for all actions.

        Arguments:
            observation: the observation for whose action count should be computed.

        Returns: the sum of action_count(observation, action) for all actions.
        """
        return np.sum([self.action_counts[observation][action] for action in self.actions])

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

        print('{} Saving model to directory: {}'.format(datetime.now(), self.model_path))

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
