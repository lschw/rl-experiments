import numpy as np

def evaluate_policy(env, pi, N_episodes, N_iterations):
    """Runs certain number of episodes using given policy and calculates
    mean return. Repeat this N_iterations times and obtain the maximum.
    """
    return_mean = []
    return_max = 0
    for i_iter in range(1, N_iterations+1):
        returns = np.zeros(N_episodes)
        for i_episode in range(N_episodes):
            done = False
            state = env.reset()
            while not done:
                action = np.argmax(pi[state])
                state,reward,done,info = env.step(action)
                returns[i_episode] += reward
        return_max = np.max([return_max, returns.max()])
        return_mean.append(returns.mean())
        print("\rIteration {:>5}".format(i_iter), end="")
        print(", mean return {:>5.2f}, max return {:>5.2f}".format(
            returns.mean(), returns.max()), end="")
        print(", best mean return so far: {:>5.2f}".format(
            np.max(return_mean)), end="")
        print(", best max return so far: {:>5.2f}".format(
            return_max), end="")

    print()
    print("Maximum return observed within {} episodes: {}".format(
        N_episodes*N_iterations, return_max))
    print("Best mean return over {} episodes within {} iterations: {}".format(
        N_episodes, N_iterations, np.max(return_mean)))
