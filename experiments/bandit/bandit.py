import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def argmax_random(values):
    """Returns random index of one of the maximum values"""
    v_max = values[0]
    v_list = []
    for i in range(len(values)):
        if values[i] > v_max:
            v_list = [i]
            v_max = values[i]
        elif values[i] == v_max:
            v_list.append(i)
    return np.random.choice(v_list)



class Bandit:
    """Represents a N-armed bandit problem"""

    def __init__(self, name="", N_arms=10, mean=0, var=1, q_init=0, alpha=0):
        """
        Args:
            name: Name of bandit problem
            N_arms: Number of arms
            mean: Mean value reward distribution
            var: Variance of reward distribution
            q_init: Initial action value
            alpha: Update step size. If 0 then alpha = 1/N
        """
        self.name = name
        self.N_arms = N_arms
        self.mean = mean
        self.var = var
        self.q_init = q_init
        self.alpha = 0
        self.reset()


    def reset(self):
        """Sets new q value distribution and reset learned parameters"""
        self.q_star = np.random.normal(self.mean, self.var, self.N_arms)
        self.q = [self.q_init]*self.N_arms
        self.N_q = np.zeros(self.N_arms)
        self.history = []


    def play(self, arm):
        """Returns random reward of playing a single arm"""
        return np.random.normal(self.q_star[arm], self.var)


    def update(self, arm, reward):
        """Update estimate of action value q for arm given reward"""
        self.N_q[arm] += 1
        alpha = self.alpha if self.alpha > 0 else 1/self.N_q[arm]
        self.q[arm] += alpha * (reward - self.q[arm])
        self.history.append((arm,reward))


    def action(self):
        raise NotImplemented()



class BanditEpsilonGreedy(Bandit):
    """N-armed bandit with epsilon-greedy action selection"""

    def __init__(self, name="", N_arms=10, mean=0, var=1, q_init=0, alpha=0,
            epsilon=0):
        super().__init__(name, N_arms, mean, var, q_init, alpha)
        self.epsilon = epsilon


    def action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.N_arms)
        else:
            return argmax_random(self.q)



class BanditUCB(Bandit):
    """N-armed bandit with upper-confidence-bound action selection"""

    def __init__(self, name="", N_arms=10, mean=0, var=1, q_init=0, alpha=0,
            c=0):
        super().__init__(name, N_arms, mean, var, q_init, alpha)
        self.c = c


    def action(self):
        t = np.sum(self.N_q) + 1 # +1 to prevent log(0)
        return argmax_random(self.q
            + self.c*np.sqrt(np.log(t)
                /(self.N_q+1e-10) # +1e-10 to prevent dividing by 0
            )
        )



if __name__ == "__main__":

    bandits = [
        BanditEpsilonGreedy(name="greedy, ɑ=1/N"),
        BanditEpsilonGreedy(name="optimistic greedy, q0=5, ɑ=1/N", q_init=5),
        BanditEpsilonGreedy(name="ε=0.1, ɑ=1/N", epsilon=0.1),
        BanditEpsilonGreedy(name="ε=0.01, ɑ=1/N", epsilon=0.01),
        BanditEpsilonGreedy(name="ε=0.1, ɑ=0.1", alpha=0.1, epsilon=0.1),
        BanditUCB(name="UCB, c=2, ɑ=1/N", c=2),
    ]

    N_runs = 2000
    N_steps = 1000
    rewards = np.zeros((len(bandits),N_steps))

    # Perform N_runs
    for run in range(N_runs):
        print("Run", run)

        # Reset bandit for each run
        for b in bandits:
            b.reset()

        # Play N_step games
        for step in range(N_steps):

            for b in bandits:
                arm = b.action()
                reward = b.play(arm)
                b.update(arm, reward)

        # Update mean reward
        for i,b in enumerate(bandits):
            rewards[i] += 1/(run+1)*(np.array(b.history)[:,1] - rewards[i])

    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(1, 1, 1)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    for i,b in enumerate(bandits):
        plt.plot(range(N_steps), rewards[i], label=b.name)
    plt.legend(loc='lower right', frameon=False)
    plt.savefig("bandit.pdf", dpi=200)

