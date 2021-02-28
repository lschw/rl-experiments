import numpy as np
import gym

class BoxDiscretization:
    """
    Discretize multidimensional gym box space

    Example:
    observation_space = Box(
        low=np.array([-1.0, -2.0]),
        high=np.array([2.0, 4.0]),
        dtype=np.float32
    )
    N_buckets = [3,4]

    Dimension 1 is discretized into 3 buckets and dimension 2 into 4 buckets,
    leading in to total 12 states.
    """

    def __init__(self, observation_space, N_buckets, limits):
        """
        Args:
            observation_space: Gym box space object
            N_buckets: Number of discretization points for each dimension, list
            limits: Limits for each dimension
        """
        if not isinstance(observation_space, gym.spaces.box.Box):
            raise TypeError("Observation space is not Box")
        self.N_buckets = N_buckets
        self.N = np.prod(N_buckets)
        self.limits = limits


    def encode(self, observation):
        """Returns discretized index of given observation

        Args:
            observation: Multidimensional observation
        """
        encoding = 0
        for i,o in enumerate(observation):

            # Normalize observation with respect to limit of range
            o_normalized = (o-self.limits[i][0]) \
                / (self.limits[i][1]-self.limits[i][0])

            # Scale according to bucket size
            o_bucket = int(o_normalized * self.N_buckets[i])
            o_bucket = min(self.N_buckets[i]-1, o_bucket)

            # Create 2d index
            encoding += o_bucket \
                * (1 if i == 0 else self.N_buckets[i-1])
        return int(encoding)
