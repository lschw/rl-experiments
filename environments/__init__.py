import sys
import os
sys.path.insert(0,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "gridworld"))
sys.path.insert(0,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "randomwalk"))

import gym_gridworld
import gym_randomwalk
