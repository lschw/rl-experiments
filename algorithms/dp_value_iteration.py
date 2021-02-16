import numpy as np
from .utils import *

def dp_value_iteration(env, gamma, tol=1e-3, iter_max=100):
    """Determines optimal policy by performing value iteration
    via Bellman optimality equation

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 83

    Args:
        env: Environment
        gamma: Discount factor
        tol: Tolerance to stop iteration
        iter_max: Maximum iteration count

    Returns
        pi: Optimal policy
        v: Value function of policy
    """
    v = np.zeros(env.observation_space.n)
    for i_iter in range(iter_max):
        print("\r> DP Value iteration: Iteration {}/{}".format(
            i_iter+1, iter_max), end="")

        delta = 0
        for state in range(env.observation_space.n):

            # Determine action-value function for state
            q = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for (prob,state2,reward,done) in env.P[state][action]:
                    q[action] += prob*(reward + gamma*v[state2])

            # Set state-value function to maximum of action-value function
            delta = max(delta, np.abs(np.max(q)-v[state]))
            v[state] = np.max(q)
        if delta < tol:
            break
    print()

    pi = np.array([utils.dp_greedy_policy(env, v, state, gamma)
        for state in range(env.observation_space.n)])
    return pi,v
