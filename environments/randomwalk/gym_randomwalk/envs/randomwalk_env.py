import numpy as np
from gym.envs.toy_text import discrete

class RandomWalkEnv(discrete.DiscreteEnv):
    """
    Description:
        A 1d random walk. Terminal states are the left and right states.
        Non-terminal transitions give a reward of 0, the reward of terminal
        transitions can be specified. The start state is in the center. At each time step, the agent moves to the left or right with equal probability within a fixed range. This is a reward process only,
        the agent cannot influence its movement.

    Observation:
        Type: Discrete(N)
        State number starting from left to right, e.g. N=10 state version
        0 1 2 3 4 5 6 7 8 9
        State 0 and 9 are terminal states, start state is 5

    Actions:
        Type: Discrete(1)
        0 : Perform random walk
    """

    def __init__(self, N=11, step_range=1, terminal_rewards=(-1,1)):
        """
        Args:
            N: Number of states
            step_range: Step range
            terminal_rewards: Rewards of the left and right terminal states
        """
        self.N = N
        self.step_range = step_range

        nA = 1
        nS = N

        # Start at center state
        isd = np.zeros(nS)
        isd[int(N/2)] = 1

        # Set transition probabilities
        P = {s: {0: []} for s in range(nS)}
        prob = 0.5/step_range
        for s in range(nS):
            if s == 0 or s == nS-1:
                continue
            for step in range(1,step_range+1):
                sleft = s-step
                sright = s+step
                if sleft >= 0:
                    P[s][0].append((
                        prob if sleft > 0 else (step_range-s+1)*prob,
                        sleft,
                        terminal_rewards[0] if sleft == 0 else 0,
                        sleft == 0
                    ))

                if sright <= nS-1:
                    P[s][0].append((
                        prob if sright < nS-1 else (s+step_range+1-nS+1)*prob,
                        sright,
                        terminal_rewards[1] if sright == nS-1 else 0,
                        sright == nS-1
                    ))

        super().__init__(nS, nA, P, isd)


    def render(self):
        """Prints current state"""
        out = ""
        for s in range(self.nS):
            out += "\033[42m" if s == self.s else ""
            out += "{}".format(s)
            out += "\033[0m " if s == self.s else " "
        print(out)
