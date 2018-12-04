import logging
import numpy as np

from collections import deque
from scipy.stats import rv_discrete, rv_continuous
from typing import Tuple, Union, Callable, TypeVar

# Define alias types for clearer documentation
History = TypeVar(np.array)
State = TypeVar(int)
Action = TypeVar(int)
Distribution = TypeVar(Union[rv_discrete, rv_continuous])

logger = logging.getLogger('lmdp.FLMDP')


class FLMDP(object):
    """Finite l-th order MDP.

    Attributes:
        mag_S (int): Cardinality of the finite state space.
        mag_A (int): Cardinality of the finite action space.
        P (Distribution): Next state/reward distribution given the current history and action,
            P(s_{t+1} | h_t, a_t). 
        P0 (Distribution): Probability distribution over initial states.
        l (int): Number of states that matter to the environment when determining the next state.
        S (np.array): Vector of integers representing the finite state space.
        A (np.array): Vector of integers representing the finite action space.

    """

    def __init__(self, 
                 mag_S: int, 
                 mag_A: int, 
                 P: Distribution,
                 P0: Distribution,
                 l: int):
        self.mag_S = mag_S
        self.mag_A = mag_A
        self.P = P
        self.P0 = P0
        self.l = l

        # 0 included for states where t < l
        self.S = np.arange(start=1, 
                           stop=mag_S+1, 
                           step=1, 
                           dtype=np.int32)

        self.A = np.arange(start=0, 
                           stop=mag_A, 
                           step=1, 
                           dtype=np.int32)


    def simulate(pi: Distribution,
                 T: int,
                 m: int):
        """Generate sample trajectories using given policy.

        Args:
            pi (Distribution): Policy distribution over actions given the current history.
            T (int): Trajectory length.
            m (int): Number of trajectories to simulate.

        Returns: 
            trajectories (np.array): Matrix of trajectories, of shape (m, T)

        """
        l = self.l
        P = self.P
        P0 = self.P0

        trajectories = np.zeros((m, T), dtype=np.float32)

        # Sample initial state for each trajectory
        trajectories[:, 0] = 1 + self.P0.rvs(size=(m, 1))

        # Generate M trajectories (s0, a0, r1, s1, a1, r2, s2, ...)
        for i in range(M):
            # Initialize history to 0 state
            h = deque((0,)*l, maxlen=l)
            for t in range(0, T, 3):
                # Update history to include the current state
                h.appendLeft(trajectories[i][t])
                # Sample action according to policy
                # TODO
                # How to give history as parameter to Distribution?
                a = pass
                trajectories[i][t+1] = a
                # Generate reward and next state
                # TODO 
                # Multidimensional values from Distribution
                s, r = pass
                trajectories[i][t+2] = r
                trajectories[i][t+3] = s
