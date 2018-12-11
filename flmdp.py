import logging
import numpy as np

from collections import deque
from scipy.stats import rv_discrete, rv_continuous
from typing import Tuple, Union, Callable, TypeVar

# Define alias types for clearer documentation
History = TypeVar(np.array)
State = TypeVar(int)
Action = TypeVar(int)
Distribution = TypeVar(np.ndarray)

logger = logging.getLogger('lmdp.FLMDP')


class FLMDP(object):
    """Finite l-th order MDP.

    Attributes:
        mag_S (int): Cardinality of the finite state space.
        mag_A (int): Cardinality of the finite action space.
        P (Distribution): Next state/reward distribution given the current history and action,
            P(s_{t+1} | h_t, a_t) = P[h_t[0], ..., h_t[l-1], a_t, s_{t+1}]
            Represented as a numpy array of shape (mag_S, ..., mag_S, mag_A, mag_S, 1)
            where len(shape)==
        P0 (Distribution): Probability distribution over initial states, so
            P(s_{0}=s) = P0[s]
            Represented as a numpy array of shape (mag_S, 1)
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
        S = self.S
        A = self.A
        l = self.l
        P = self.P
        P0 = self.P0

        # s_t from t=0, ..., T
        s_t = np.zeros((m, T+1), dtype=np.float32)
        # a_t from t=0, ..., T-1
        a_t = np.zeros((m, T), dtype=np.float32)
        # r_t from t=1, ..., T
        r_t = np.zeros((m, T), dtype=np.float32)

        s_t[:, 0] = 1 + np.random.choice(S, size=(m, 1))

        # Generate M trajectories (s0, a0, r1, s1, a1, r2, s2, ...)
        for i in range(M):
            # Initialize history to 0 state
            h = deque((0,)*l, maxlen=l)
            for t in range(0, T):
                # Update history to include the current state
                h.appendLeft(s_t[t])
                # Sample action according to policy
                a_dist = pi[h, ...]
                a = np.random.choice(A, p=a_dist)
                a[t] = a
                # Generate reward and next state
                P[h, a

                r[t] = r
                s[t+1] = s
