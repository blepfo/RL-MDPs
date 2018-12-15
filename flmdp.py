import itertools as it
import logging
from collections import deque

import numpy as np

from _types import Distribution
from policy_approximators import sparsity_corrected_approx

logger = logging.getLogger('lmdp.FLMDP')


class FLMDP(object):
    """Finite l-th order MDP.

    Attributes:
        mag_S (int): Cardinality of the finite state space.
        mag_A (int): Cardinality of the finite action space.
        P (Distribution): Next state/reward distribution given the current history and action,
            P(s_{t+1} | h_t, a_t) = P[h_t[0], ..., h_t[l-1], a_t, s_{t+1}]
            Represented as a numpy array of shape (mag_S, ..., mag_S, mag_A, mag_S, 2)
            so that p = P[tuple(h)+(a,)][i] is an array of shape (2, 1) so that
                * p[0] = probability next state is state i
                * p[1] = reward associated with transitioning to the next state
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

    def simulate(self,
                 pi: Distribution,
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
        s_t = np.zeros((m, T+1), dtype=np.int32)
        # a_t from t=0, ..., T-1
        a_t = np.zeros((m, T), dtype=np.int32)
        # r_t from t=1, ..., T
        r_t = np.zeros((m, T), dtype=np.float32)

        # Initial states for each trajectory
        s_t[:, 0] = np.random.choice(S, size=(m,))

        # Generate m trajectories
        for i in range(m):
            # Initialize history to 0 state
            h = deque((0,)*l, maxlen=l)
            for t in range(0, T):
                # Update history to include the current state
                h.appendleft(s_t[i, t])
                # Sample action according to policy
                a_dist = pi[tuple(h)]
                a = np.random.choice(A, p=a_dist)
                # Generate reward and next state
                p = P[tuple(h)+(a,)]
                # p[:, 0] is distribution over next states
                # p[i, 1] is reward for transitioning to state i
                s = np.random.choice(S, p=p[:, 0])
                r = p[s-1][1]
                # Save current time step
                s_t[i, t+1] = s
                a_t[i, t] = a
                r_t[i, t] = r

        return s_t, a_t, r_t

    @staticmethod
    def random_P(mag_S: int,
                 mag_A: int,
                 l: int,
                 mean_reward: float):
        """Generate random next state/reward distribution tensor.

        Args:
            mag_S (int): Cardinality of the finite state space.
            mag_A (int): Cardinality of the finite action space.
            l (int): Number of states that matter to the environment when determining the next state

        Returns:
            P (Distribution): Next state/reward distribution given the current history and action,
                P(s_{t+1} | h_t, a_t) = P[h_t[0], ..., h_t[l-1], a_t, s_{t+1}]
                Represented as a numpy array of shape (mag_S, ..., mag_S, mag_A, mag_S, 2)
                so that p = P[tuple(h)+(a,)][i] is an array of shape (2, 1) so that
                    * p[0] = probability next state is state i
                    * p[1] = reward associated with transitioning to the next state

        """
        shape = ((mag_S+1,)*l) + (mag_A, mag_S, 2)
        P = np.random.random(size=shape)

        # Generate tuples to access p=P[(tuple(h)+(a,)]
        history_sizes = it.repeat(list(range(mag_S+1)), l)
        dist_sizes = it.chain(history_sizes, [list(range(mag_A))])
        history_actions = it.product(*dist_sizes)

        for history_action in history_actions:
            # Normalize the next state distribution
            p = P[tuple(history_action)]
            p[:, 0] = softmax(p[:, 0])
            # Use positive and negative reward values
            for i in range(mag_S):
                if np.random.random() > 0.5:
                    p[i, 1] *= -1
                    p[i, 1] += mean_reward

        return P

    @staticmethod
    def random_pi(lmdp) -> Distribution:
        """Generate random policy tensor.

        Args:
            lmdp (FLMDP): L-MDP to create the policy for.

        pi (Distribution): Policy distribution over actions given the current history.

        """
        mag_S = lmdp.mag_S
        mag_A = lmdp.mag_A
        l = lmdp.l

        shape = ((mag_S+1,)*l) + (mag_A,)
        pi = np.random.random(size=shape)

        # Generate tuples to access pi[(tuple(h)]
        history_sizes = it.repeat(list(range(mag_S+1)), l)
        histories = it.product(*history_sizes)

        for history in histories:
            # Normalize the next action distribution
            pi[tuple(history)] = softmax(pi[tuple(history)])

        return pi

    @staticmethod
    def scips_approximable_pi(lmdp,
                              Gamma: float,
                              sigma: float,
                              T=100,
                              m=1000) -> Distribution:
        """Generate policy tensor under SCIPS assumption.

        Args:
            lmdp (FLMDP): FLMDP for which to make SCIPS approximable pi.
            Gamma (float): Time discounting parameter used in the SCIPS,
                in [0.0, 1.0].
            sigma (float): TODO
            T (int): Trajectory length.
            m (int): Number of trajectories to simulate.

        pi (Distribution): Policy distribution over actions given the current history.

        """
        mag_S = lmdp.mag_S
        mag_A = lmdp.mag_A
        l = lmdp.l

        # Start with a random pi
        pi = FLMDP.random_pi(lmdp=lmdp)

        # Simuate some trajectories
        s_t, r_t, a_t = lmdp.simulate(pi=pi,
                                      T=T,
                                      m=m)

        # Fit pi to SCIPS
        scips = sparsity_corrected_approx(states=s_t,
                                          actions=a_t,
                                          rewards=r_t,
                                          Gamma=Gamma,
                                          lmdp=lmdp)

        for history_action in scips:
            scips[history_action] += np.random.normal(loc=0, scale=sigma)

        # Normalize the next action distribution
        history_sizes = it.repeat(list(range(lmdp.mag_S+1)), lmdp.l)
        histories = it.product(*history_sizes)

        for history in histories:
            pi[tuple(history)] = l1_normalize(pi[tuple(history)])

        return pi


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def l1_normalize(x):
    return x / x.sum()


if __name__ == "__main__":
    mag_S = 25
    mag_A = 4
    l = 3
    T = 100

    # Create example MDP
    P0 = (1.0 / float(mag_S)) * np.ones((mag_S))
    P = FLMDP.random_P(mag_S=mag_S,
                       mag_A=mag_A,
                       l=l,
                       mean_reward=100)
    lmdp = FLMDP(mag_S=mag_S,
                 mag_A=mag_A,
                 P=P,
                 P0=P0,
                 l=l)

    # Simulate trajectories with example policy
    pi = FLMDP.random_pi(lmdp=lmdp)

    s_t, r_t, a_t = lmdp.simulate(pi=pi,
                                  T=100,
                                  m=1000)

    print(s_t)
    print(r_t)
    print(a_t)
