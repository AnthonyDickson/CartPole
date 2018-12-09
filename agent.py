from datetime import datetime
import os

import numpy as np
import pickle
from gym import make
from gym.spaces import Discrete, Box

from utils.bucketing import MultiBucketer
from utils.datastructures import ObservationDict
from utils.path import get_run_path

class CartPoleAgent:
    """A Q-Learning agent for the cart-pole problem.
    
    The observation space for the cart pole problem is continuous so the agent buckets (discretises) the observation data.
    """
    def __init__(self, action_space: Discrete, observation_space: Box, n_buckets: int=100, learning_rate=0.1, learning_rate_annealing=None,
        discount_factor=0.99, exploration_rate=1.0, exploration_rate_annealing=None, initial_q_value = 0, input_mask=None):
        """Setup the agent.

        Arguments:
            action_space: The action space of the cart-pole enviroment.
            observation_space: the observation space of the cart-pole enviroment.
            n_buckets: how many buckets to separate the observation space into (since it is continuous).
            learning_rate: how much a recent experience affects future actions.
            learning_rate_annealing: an Annealer object that decays the learning rate over time.
            discount_factor: how much weight future Q-values have during Q-value update (also called gamma).
            exploration_rate: the C variable from the bonus function as defined in UCB-1. Controls how much exploration is done.
            exploration_rate_annealing: an Annealer object that decays the exploration rate over time.
            initial_q_value: the value the Q-values should be initialised to.
            input_mask: a binary mask as a list of integers, with 0 indicating the value should be ignored and 1 indicating the value should be left untouched.
        """
        self.bucketer = MultiBucketer(observation_space.low, observation_space.high, n_buckets)
        self.actions = np.arange(0, action_space.n)
        self.action_counts = ObservationDict(0, action_space.n)
        self.q_table = ObservationDict(initial_q_value, action_space.n)
        self.learning_rate = learning_rate
        self.learning_rate_annealing = learning_rate_annealing
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_rate_annealing = exploration_rate_annealing

        assert input_mask is None or len(input_mask) == self.bucketer.n  # ensure input mask is same dimensions as observation (state) space.
        self.input_mask = input_mask if input_mask else np.ones(self.bucketer.n)

        self.model_path = get_run_path(prefix='data/')

    def get_action(self, observation, t=0):
        """Get the best action based on the current observation.
        
        Arguments:
            observation: a set of observation values from the environment.
            t: the timestep.

        Returns: the optimal action (integer) based on the current q_table
        """
        observation *= self.input_mask
        bucketed = self.bucketer.get_bucketed(observation)

        # UCB-1 first chooses any actions that have yet to be chosen at least once.
        for action in self.actions:
            if self.action_counts[bucketed][action] < 1:
                self.action_counts[bucketed][action] = 1

                return action

        q_values = [self.q_table[bucketed][action] + self.bonus(bucketed, action, t) for action in self.actions]
        action = np.argmax(q_values)
        self.action_counts[bucketed][action] += 1

        return action

    def update(self, prev_observation, prev_action, reward, observation, t=0):
        """Update the Q-value for the previous observation and action.
        
        Arguments:
            prev_observation: the set of observation values from the previous step.
            prev_action: the action taken last step.
            reward: the reward from taking the previous action.
            observation: the set of observation values for the next step.
            t: the timestep in the current episode.
        """
        prev_observation *= self.input_mask
        observation *= self.input_mask
        prev_bucketed = self.bucketer(prev_observation)
        bucketed = self.bucketer(observation)

        prev_Q = self.q_table[prev_bucketed][prev_action]
        action = np.argmax(self.q_table[bucketed])
        next_Q = self.q_table[bucketed][action]

        if self.learning_rate_annealing:
            a = self.learning_rate_annealing(self.learning_rate, t)
        else:
            a = self.learning_rate
        
        g = self.discount_factor
        self.q_table[prev_bucketed][prev_action] = (1 - a) * prev_Q  + a * (reward + g * next_Q)

    def bonus(self, observation, action, t):
        """Calculate the exploration bonus for the observation-action pair.

        Based on the UCB-1 exploration method where the bonus is defined as:

            bonus(st, ai) = 100 × C × √(2 × log N(st) / N(st, ai)

        where st is the state (or observation) at timestep t,
            ai is the an action from the action space,
            C is a constant that increases the amount of exploration,
            N(st) is the number of times an action was chosen for the chosen state,
            N(st, ai) is the number of times an action was chosen under the chosen state.

        Arguments:
            observation: an observation from the environment.
            action: the action to be taken.
            t: the timestep used for annealing.

        Returns: the exploration bonus (to the Q-value) for taking the given action and observation.
        """
        N_st = self.observation_count(observation)
        N_st_ai = self.action_counts[observation][action]

        if self.exploration_rate_annealing:
            C = self.exploration_rate_annealing(self.exploration_rate, t)
        else:
            C = self.exploration_rate

        return 100 * C * np.sqrt(2 * np.log(N_st) / N_st_ai)

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
        os.makedirs(self.model_path, exist_ok=True)
        path = self.model_path + filename

        with open(path, 'wb') as f:
            f.write(pickle.dumps(self))

        print('[{}] Saving model to: {}'.format(datetime.now(), path))

        return path

    @staticmethod
    def load(fullpath):
        """Load a saved model.

        Arguments:
            fullpath: the path including the filename of the saved model.

        Returns: the saved model at the designated path.
        """
        with open(fullpath, 'rb') as f:
            return pickle.load(f)    
