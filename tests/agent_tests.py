import os
import sys
import unittest
sys.path.append(os.getcwd())

from gym import make

from agent import CartPoleAgent


def test(test_fn):
    def wrapper(*args):
        self = args[0]

        self.setup()
        test_fn(self)
        self.teardown()

    return wrapper

class TestAgent(unittest.TestCase):
    def setup(self):
        self.env = make('CartPole-v0')
        self.agent = CartPoleAgent(self.env.action_space, self.env.observation_space)

    def teardown(self):
        self.env.close()
        del self.agent

    @test
    def test_bucketing(self):
        observation = self.env.reset() 

        assert self.agent.get_bucketed(observation).min() != -1
        
        for value in self.agent.get_bucketed(observation):
            assert 0 <= value <= self.agent.n_buckets

    @test
    def test_valid_selection(self):
        observation = self.env.reset()
        
        action = self.agent.get_action(observation)
        assert self.env.action_space.contains(action)

    @test
    def test_updates_q_values(self):
        observation = self.env.reset()        
        action = self.agent.get_action(observation)
        
        prev_q_table = str(self.agent.q_table)    
        prev_observation = observation
        prev_action = action

        observation, reward, _, _ = self.env.step(action)

        self.agent.update(prev_observation, prev_action, reward, observation)
        assert str(self.agent.q_table) != prev_q_table, 'Q table unchanged:\n{}\nVS\n{}'.format(self.agent.q_table, prev_q_table)

    @test
    def test_model_saving(self):
        observation = self.env.reset()        
        action = self.agent.get_action(observation)        
        observation, reward, _, _ = self.env.step(action) 
        prev_observation = observation
        prev_action = action

        self.agent.update(prev_observation, prev_action, reward, observation)

        self.agent.save()
        model_path = self.agent.model_path
        prev_q_table = self.agent.q_table
        del self.agent

        self.agent = CartPoleAgent.load(model_path)
        assert str(self.agent.q_table) == str(prev_q_table)

if __name__ == '__main__':
    unittest.main()