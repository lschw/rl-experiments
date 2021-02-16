import numpy as np
from gym.envs.toy_text import discrete

class GridworldEnv(discrete.DiscreteEnv):
    """
    Description:
        A 2d rectangular grid world where agent can move deterministically
        up, down, left, right. Each step gives a fixed reward. Some grid cells
        can be declared as terminal states. Moves outside of grid set
        agent to previous cell.

    Observation:
        Type: Discrete(N)
        Grid cell number starting from top-left to bottom-right, e.g. 4x4 grid
        0   1  2  3
        4   5  6  7
        8   9 10 11
        12 13 14 15

    Actions:
        Type: Discrete(4)
        0 : Move down
        1 : Move up
        2 : Move right
        3 : Move left
    """

    def __init__(self,
            grid=np.array([
                [-1,-1,-1,-1,],
                [-1,-1,-1,-1,],
                [-1,-1,-1,-1,],
                [-1,-1,-1,-1],
            ]), terminal_states=[0, 15]):
        """
        Args:
            grid: 2d array representing grid with immediate rewards
            terminal_states: List of terminal states
        """
        self.Nx = grid.shape[1]
        self.Ny = grid.shape[0]
        self.grid = grid
        self.terminal_states = terminal_states

        nA = 4
        nS = self.Nx*self.Ny

        # All states except terminal states get equal start probability
        isd = np.ones(nS) / (nS-len(terminal_states))
        isd[self.terminal_states] = 0

        # Set transition probabilities
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for y in range(self.Ny):
            for x in range(self.Nx):
                s = y*self.Nx + x

                if s in terminal_states:
                    continue

                for a in range(nA):
                    # Perform action as 2d vector operation on grid
                    s_vec = np.array([y,x])
                    a_vec = [[1,0],[-1,0],[0,1],[0,-1]][a]
                    s2_vec = s_vec + a_vec
                    y2,x2 = s2_vec[0],s2_vec[1]

                    if y2 < 0 or x2 < 0 or y2 >= self.Ny or x2 >= self.Nx:
                        # Move outside of grid -> original state
                        y2 = y
                        x2 = x

                    s2 = y2*self.Nx + x2
                    P[s][a].append((1,s2,grid[y2,x2],s2 in terminal_states))

        super().__init__(nS, nA, P, isd)


    def render(self):
        """Prints current state in 2d grid"""
        w = 3
        out = "+" + "-"*(self.Nx*(w+2) + (self.Nx-1)) + "+\n"
        for y in range(self.Ny):
            row = []
            for x in range(self.Nx):
                s = y*self.Nx + x
                cell = "\033[42m" if s == self.s else ""
                cell += "T" if s in self.terminal_states else " "
                cell += "{:>3}".format(self.grid[y,x])
                cell += "*\033[0m" if s == self.s else " "
                row.append(cell)

            out += "|" + ":".join(row) + "|\n"
        out += "+" + "-"*(self.Nx*(w+2) + (self.Nx-1)) + "+\n"
        print(out)
