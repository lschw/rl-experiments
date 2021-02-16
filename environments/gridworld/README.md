# 2d Grid World

This is an OpenAI Gym environment for a two-dimensional rectangular grid world
where an agent can move deterministically up, down, left, right. Each step
gives a fixed reward. Some grid cells can be declared as terminal states.
A move outside of the grid does not change the position.


## Installation

Install via

    cd /path/to/source
    pip install .

Uninstall via

    pip uninstall gym-gridworld


## Usage

    import gym
    import gym_gridworld
    env = gym.make("Gridworld-v0", grid=np.array([
        [ 1,-1,-1,-1,],
        [-1,-1,-1,-1,],
        [ 1,-1, 5,-1,],
        [-1,-1,-1,-1,],
        [-1,-1,-1,-1,],
    ]), terminal_states=[0, 15])
