import os
import sys
import unittest
sys.path.append(os.getcwd())

from utils.datastructures import ObservationDict
from utils.bucketing import MultiBucketer

class TestObservationDict(unittest.TestCase):
    def test_can_add_values(self):
        d = ObservationDict(0, 2, None)

        idx = [0, 1]
        d[idx][0] = 1
        assert d[idx][0] == 1

    def test_can_modify_values(self):
        a = 0.123
        b = 0.321

        bucketer = MultiBucketer([0, 0], [1.0, 1.0], 10)
        d = ObservationDict(0, 2, bucketer)

        idx = [a, b]
        d[idx][0] = 1
        assert d[idx][0] == 1

        d[idx][0] = 3
        assert d[idx][0] == 3

        array = d[idx]
        array[1] = 1

        assert d[idx][1] == 1

if __name__ == '__main__':
    unittest.main()