import numpy as np

def evaluate_policy(env, pi, N_episodes):
    """Runs certain number of episodes using given policy and calculates
    mean return.
    """
    returns = np.zeros(N_episodes)
    for i_episode in range(N_episodes):
        done = False
        state = env.reset()
        while not done:
            action = np.argmax(pi[state])
            state,reward,done,info = env.step(action)
            returns[i_episode] += reward
        print("\rEpisode {:>5}".format(i_episode+1), end="")
        print(", return {:>7.2f}".format(
           returns[i_episode]), end="")
        print(", mean return so far: {:>7.2f}".format(
            np.mean(returns)), end="")
        print(", max return so far: {:>7.2f}".format(
            np.max(returns)), end="")

    print()
    print("Maximum return observed within {} episodes: {}".format(
        N_episodes, np.max(returns)))
    print("Mean return over {} episodes: {}".format(
        N_episodes, np.mean(returns)))
