import numpy as np
from .utils import *

def dp_policy_evaluation(env, pi, v=None, gamma=1, tol=1e-3, iter_max=100,
        verbose=True):
    """Evaluates state-value function by performing iterative policy evaluation
    via Bellman expectation equation (in-place)

    Based on Sutton/Barto, Reinforcement Learning, 2nd ed. p. 75

    Args:
        env: Environment
        pi: Policy
        v: Initial value function or None
        gamma: Discount factor
        tol: Tolerance to stop iteration
        iter_max: Maximum iteration count

    Returns:
        v: State-value function
    """
    if v is None:
        v = np.zeros(env.observation_space.n)
    for i_iter in range(iter_max):
        if verbose:
            print("\r> DP Policy evaluation: Iteration {}/{}".format(
                i_iter+1, iter_max), end="")
        delta = 0
        for state in range(env.observation_space.n):
            v_new = 0
            for action in range(env.action_space.n):
                for (prob,state2,reward,done) in env.P[state][action]:
                    v_new += pi[state][action] * prob * (
                        reward + gamma*v[state2]
                    )
            delta = max(delta, np.abs(v_new-v[state]))
            v[state] = v_new
        if delta < tol:
            break
    if verbose:
        print()
    return v
