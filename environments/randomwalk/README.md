# Random Walk

This is a 1d random walk OpenAI Gym environment.
Terminal states are the left and right states.
Non-terminal transitions give a reward of 0,
the reward of terminal transitions can be specified.
The start state is in the center.
At each time step, the agent moves to the left or right
with equal probability within a fixed range.
This is a reward-only process, the agent cannot influence its movement.

## Installation

Install via

    cd /path/to/source
    pip install .

Uninstall via

    pip uninstall gym-randomwalk


## Usage

    import gym
    import gym_randomwalk
    env = gym.make("RandomWalk-v0",
        N=1000, step_range=100, terminal_rewards=(-1,1))
