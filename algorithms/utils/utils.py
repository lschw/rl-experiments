import numpy as np
from collections import defaultdict


def select_action_greedy(Q, s):
    return np.argmax(Q[s])


def select_action_epsilon_greedy(Q, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q[s]))
    else:
        return np.argmax(Q[s])


def select_action_random(Q, s):
    return np.random.choice(len(Q[s]))


def select_action_policy(pi, s):
    return np.random.choice(len(pi[s]), p=pi[s])


def create_epsilon_soft_policy_state(Q, s, epsilon):
    return [1 - epsilon + epsilon/len(Q[s])
        if a == np.argmax(Q[s])
        else epsilon/len(Q[s])
        for a in range(len(Q[s]))
    ]

def create_greedy_policy_state(Q, s):
    return create_epsilon_soft_policy_state(Q, s, 0)


def create_epsilon_soft_policy(Q, epsilon):
    pi = defaultdict(lambda: np.ones(len(Q[0]))/len(Q[0]))
    for s in Q:
        pi[s] = create_epsilon_soft_policy_state(Q, s, epsilon)
    return pi


def create_greedy_policy(Q):
    return create_epsilon_soft_policy(Q, 0)


def create_state_value_function(Q, pi):
    v = defaultdict(lambda: 0)
    for s in pi:
        for a in range(len(Q[s])):
            v[s] += pi[s][a]*Q[s][a]
    return v


def decay_none(v, i, N, vmin=0.01):
    return v


def decay_linear(v, i, N, vmin=0.01):
    return v*(1 + (vmin-1)*i/N)


def decay_exponential(v, i, N, vmin=0.01):
    return v*vmin**(i/N)


def decay_sigmoid(v, i, N, vmin=0.01):
    a = 10
    b = 1 + np.log(vmin/(1-vmin))/10
    return v/(1+np.exp(a*(i/N-b)))


def random_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def generate_episode(env, pi, exploring_starts=False, ep_max_length=1000):
    """Generates episode following policy

    Args:
        env: Environment
        pi: Policy
        exploring_starts: Whether to use exploring starts
        ep_max_length: Force termination of episode after this number of steps
                       to prevent infinite episodes resulting from bad policies

    Returns:
        episode: List of (state,action,reward)
    """
    episode = []
    done = False
    state = env.reset()
    if exploring_starts:
        action = env.action_space.sample()
    else:
        action = select_action_policy(pi, state)
    while not done and len(episode) < ep_max_length:
        state_new, reward, done, info = env.step(action)
        episode.append((state,action,reward))
        state = state_new
        action = select_action_policy(pi, state)
    return episode


def dp_greedy_policy(env, v, state, gamma):
    """Returns greedy policy for state according to state-value function v
        and model of environment

    Args:
        env: Environment
        v: State-value function
        state: State
        gamma: Discount factor

    Returns:
        Policy
    """
    # Determine action-value function for state
    q = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for (prob,state2,reward,done) in env.P[state][action]:
            q[action] += prob*(reward + gamma*v[state2])

    # Give all actions with highest action-value function equal probability,
    # all other actions get zero probability
    qmax = np.max(q)
    qmax_cnt = np.sum(q == qmax)
    return [1/qmax_cnt if x else 0 for x in (q == qmax)]

