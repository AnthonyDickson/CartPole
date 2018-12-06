import numpy as np
from gym import make
from gym.spaces import Discrete, Box

class CartPoleAgent:
    """A Q-Learning agent for the cart-pole problem.
    
    The observation space for the cart pole problem is continuous so the agent buckets (discretises) the observation data.
    """
    def __init__(self, action_space: Discrete, observation_space: Box, n_buckets: int=100):
        """Setup the agent.

        Arguments:
            action_space: The action space of the cart-pole enviroment.
            observation_space: the observation space of the cart-pole enviroment.
            n_buckets: how many buckets to separate the observation space into (since it is continuous).
        """
        self.actions = np.arange(0, action_space.n)
        self.n_buckets = n_buckets
        self.low = observation_space.low
        self.high = observation_space.high

    def get_action(self, observation):
        return 0

    def get_buckets(self, observations):
        """Get the buckets for each observation.
        
        Arguments:
            observations: the set of observations from the environment.

        Returns: the observations as bucketed values.
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

if __name__ == "__main__":
    env = make('CartPole-v0')
    agent = CartPoleAgent(env.action_space, env.observation_space)

    observation = env.reset() 
    print('observation:', observation)
    print('bucketed observation({} buckets): {}'.format(agent.n_buckets, agent.get_buckets(observation)))

    assert agent.get_buckets(observation).min() != -1

    env.close()
