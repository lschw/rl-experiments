import sys
sys.path.insert(0, "../")
import numpy as np
import gym
import gym_gridworld
import unittest

DOWN = 0
UP = 1
RIGHT = 2
LEFT = 3


class GridworldTest(unittest.TestCase):

    def setUp(self):
        self.env = gym.make("Gridworld-v0", grid=np.array([
            [1,-1,-1,-1,],
            [-1,-1,5,-1,],
            [-1,-1,-1,-1,],
            [-1,-1,-1,-1,],
            [-1,-1,-1,-1,],
        ]), terminal_states=[0, 15])
        np.random.seed(1)
        self.env.seed(1)
        self.env.action_space.seed(1)
        self.env.observation_space.seed(1)
        self.env.reset()


    def test_environment(self):
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.observation_space.n, 20)
        self.assertEqual(self.env.nA, 4)
        self.assertEqual(self.env.nS, 20)
        self.assertEqual(self.env.Nx, 4)
        self.assertEqual(self.env.Ny, 5)
        self.assertEqual(self.env.isd[0], 0)
        self.assertEqual(self.env.isd[15], 0)
        self.assertEqual(self.env.isd[1], 1/18.)


    def test_P(self):
        self.assertEqual(self.env.P[0][DOWN], [])
        self.assertEqual(self.env.P[15][DOWN], [])
        self.assertEqual(self.env.P[1][DOWN], [(1,5,-1,False)])
        self.assertEqual(self.env.P[1][UP], [(1,1,-1,False)])
        self.assertEqual(self.env.P[1][RIGHT], [(1,2,-1,False)])
        self.assertEqual(self.env.P[1][LEFT], [(1,0,1,True)])
        self.assertEqual(self.env.P[5][DOWN], [(1,9,-1,False)])
        self.assertEqual(self.env.P[5][UP], [(1,1,-1,False)])
        self.assertEqual(self.env.P[5][RIGHT], [(1,6,5,False)])
        self.assertEqual(self.env.P[5][LEFT], [(1,4,-1,False)])
        self.assertEqual(self.env.P[16][DOWN], [(1,16,-1,False)])
        self.assertEqual(self.env.P[16][UP], [(1,12,-1,False)])
        self.assertEqual(self.env.P[16][RIGHT], [(1,17,-1,False)])
        self.assertEqual(self.env.P[16][LEFT], [(1,16,-1,False)])


    def test_step(self):
        self.env.s = 1
        self.assertEqual(self.env.step(DOWN), (5,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 5)
        self.env.s = 1
        self.assertEqual(self.env.step(UP), (1,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 1)
        self.env.s = 1
        self.assertEqual(self.env.step(RIGHT), (2,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 2)
        self.env.s = 1
        self.assertEqual(self.env.step(LEFT), (0,1,True,{"prob": 1}))
        self.assertEqual(self.env.s, 0)

        self.env.s = 18
        self.assertEqual(self.env.step(DOWN), (18,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 18)
        self.env.s = 18
        self.assertEqual(self.env.step(UP), (14,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 14)
        self.env.s = 18
        self.assertEqual(self.env.step(RIGHT), (19,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 19)
        self.env.s = 18
        self.assertEqual(self.env.step(LEFT), (17,-1,False,{"prob": 1}))
        self.assertEqual(self.env.s, 17)

        self.env.s = 0
        with self.assertRaises(ValueError) as cm:
            self.env.step(DOWN)


    def test_z_render(self):
        print()
        self.env.render()
        self.env.s = 0
        self.env.render()


if __name__ == "__main__":
    unittest.main(verbosity=2)
