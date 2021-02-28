import numpy as np

class StateAggregation:
    """Combine multiple states into groups
    and provide linear feature vector for function approximation"""

    def __init__(self, N_states, group_size, N_actions=1):
        """
        Args:
            N_states: Total number of states
            group_size: Combine this many states into group
            N_actions: Number of actions
        """
        self.N_states = N_states
        self.N_actions = N_actions
        self.group_size = group_size
        self.size = int(self.N_states/self.group_size)*N_actions
        self.index = {}
        for s in range(self.N_states):
            for a in range(self.N_actions):
                self.index[(s,a)] = int(s/self.group_size) \
                    + a*int(self.N_states/self.group_size)

    def q(self, state, action, w):
        """Returns action-value function

        Args:
            state: State index
            action: Action index
            w: Weight vector
        """
        return w[self.index[(state,action)]]

    def q_deriv(self, state, action, w):
        """Returns gradient of action-value function with respect to weights

        Args:
            state: State index
            action: Action index
            w: Weight vector
        """
        feature_vector = np.zeros(self.size)
        feature_vector[self.index[(state,action)]] = 1
        return feature_vector

    def v(self, state, w):
        """Returns state-value function

        Args:
            state: State index
            w: Weight vector
        """
        return self.q(state, 0, w)

    def v_deriv(self, state, w):
        """Returns gradient of state-value function with respect to weights

        Args:
            state: State index
            w: Weight vector
        """
        return self.q_deriv(state, 0, w)

    def generate_weights(self):
        """
        Returns weight vector
        """
        return np.zeros(self.size)
