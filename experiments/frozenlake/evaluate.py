import sys
sys.path.insert(0, "../../")
import algorithms as alg
import numpy as np

def evaluate_policy(env, pi, N_episodes, goal_state):
    """Runs certain number of episodes using given policy and evaluates
    how often the goal state is reached
    """
    alg.utils.random_seed(env, 1)
    end_states = np.zeros(N_episodes, dtype=int)
    steps = np.zeros(N_episodes, dtype=int)
    for i_episode in range(N_episodes):
        print("\rEvaluate episode {}/{}".format(i_episode, N_episodes), end="")
        done = False
        state = env.reset()
        while not done:
            action = alg.utils.select_action_policy(pi, state)
            state,reward,done,info = env.step(action)
            steps[i_episode] += 1
        end_states[i_episode] = state

    print()
    print("Goal reached in {:.2f}% of episodes".format(
        np.sum(end_states == goal_state)/N_episodes*100))
    print("Shortest path length: {}, average path length: {:.2f}".format(
        steps.min(), steps.mean()))
    return np.sum(end_states == goal_state)/N_episodes


def td_hyperparameter_search(algorithm, env):
    """Performs hyperparameters search for TD algorithms"""
    import itertools
    import operator

    list_gamma = [1,0.99,0.95,0.9,0.8]
    list_alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    list_alpha_decay = [alg.utils.decay_none,alg.utils.decay_linear,
        alg.utils.decay_sigmoid]
    list_epsilon = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    list_epsilon_decay = [alg.utils.decay_none,alg.utils.decay_linear,
        alg.utils.decay_sigmoid]

    params = list(itertools.product(list_gamma,list_alpha,list_alpha_decay,
        list_epsilon,list_epsilon_decay))

    result = {}
    for p in params:
        alg.utils.random_seed(env, 1)
        gamma,alpha,alpha_decay,epsilon,epsilon_decay = p
        Q,history = algorithm(
            env, alpha=alpha, gamma=gamma, epsilon=epsilon, N_episodes=10000,
            epsilon_decay=epsilon_decay, alpha_decay=alpha_decay)
        pi = alg.utils.create_greedy_policy(Q)

        result[p] = evaluate_policy(env, pi, 10000, env.nS-1)
        p_max = max(result.items(), key=operator.itemgetter(1))[0]
        print("Current: {}\n -> Result: {}\nBest: {}\n -> Result: {}\n".format(
            p, result[p], p_max, result[p_max])
        )

    return result
