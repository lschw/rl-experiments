import sys
sys.path.insert(0, "../")
import numpy as np
import gym
import gym_randomwalk
import unittest

class RandomWalkTest(unittest.TestCase):

    def setUp(self):
        self.env = gym.make("RandomWalk-v0", N=7, step_range=4)
        np.random.seed(1)
        self.env.seed(1)
        self.env.action_space.seed(1)
        self.env.observation_space.seed(1)
        self.env.reset()


    def test_environment(self):
        self.assertEqual(self.env.action_space.n, 1)
        self.assertEqual(self.env.observation_space.n, 7)
        self.assertEqual(self.env.nA, 1)
        self.assertEqual(self.env.nS, 7)
        self.assertEqual(self.env.isd[0], 0)
        self.assertEqual(self.env.isd[6], 0)
        self.assertEqual(self.env.isd[3], 1)


    def test_P(self):
        print(self.env.P[0][0])
        self.assertEqual(self.env.P[0][0], [])
        self.assertEqual(self.env.P[6][0], [])
        self.assertEqual(self.env.P[1][0], [
            (0.5,0,-1,True),
            (0.125,2,0,False),
            (0.125,3,0,False),
            (0.125,4,0,False),
            (0.125,5,0,False),
        ])
        self.assertEqual(self.env.P[2][0], [
            (0.125,1,0,False),
            (0.125,3,0,False),
            (0.375,0,-1,True),
            (0.125,4,0,False),
            (0.125,5,0,False),
            (0.125,6,1,True),
        ])
        self.assertEqual(self.env.P[4][0], [
            (0.125,3,0,False),
            (0.125,5,0,False),
            (0.125,2,0,False),
            (0.375,6,1,True),
            (0.125,1,0,False),
            (0.125,0,-1,True),
        ])
        self.assertEqual(self.env.P[5][0], [
            (0.125,4,0,False),
            (0.5,6,1,True),
            (0.125,3,0,False),
            (0.125,2,0,False),
            (0.125,1,0,False),
        ])


    def test_z_render(self):
        print()
        self.env.render()
        self.env.s = 0
        self.env.render()


if __name__ == "__main__":
    unittest.main(verbosity=2)
