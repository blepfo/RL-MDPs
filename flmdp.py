""" Defines a finite l-th order MDP. """

from collections import deque

import numpy as np

from _types import Distribution, Trajectory, Policy
from policy_approximators import sparsity_corrected_approx
from utils import l1_normalize, policy_shape, history_tuples, history_action_tuples, transitions_shape
from scipy.special import softmax


class FLMDP(object):
    """Finite l-th order MDP.

    Attributes:
        state_size (int): Cardinality of the finite state space.
        action_size (int): Cardinality of the finite action space.
        transition_probability (Distribution): Next state/reward distribution given the current
            history and action, P(s_{time+1} | h_t, a_t) =
            transition_probability[h_t[0], ..., h_t[history_length-1], a_t, s_{time+1}]
            Represented as a numpy array of shape
            (state_size, ..., state_size, action_size, state_size, 2)
            where p = transition_probability[tuple(history)+(a,)][i] is an array of shape (2, 1)
            where
                * p[0] = probability next state is state i
                * p[1] = reward associated with transitioning to the next state
        initial_state_probability (Distribution): Probability distribution over initial states, so
            P(s_{0}=state) = initial_state_probability[state]
            Represented as a numpy array of shape (state_size, 1)
        history_length (int): Number of states that matter to the environment when determining the
            next state.
        states (np.array): Vector of integers representing the finite state space.
        actions (np.array): Vector of integers representing the finite action space.

    """

    def __init__(self, state_size: int, action_size: int,
                 transition_probability: Distribution,
                 initial_state_probability: Distribution, history_length: int):
        self.state_size = state_size
        self.action_size = action_size
        self.transition_probability = transition_probability
        self.initial_state_probability = initial_state_probability
        self.history_length = history_length

        # 0 included for states where time < history_length
        self.states = np.arange(
            start=0, stop=state_size, step=1, dtype=np.int32)

        self.actions = np.arange(
            start=0, stop=action_size, step=1, dtype=np.int32)

    def simulate(self, policy: Policy, time_horizon: int,
                 n_samples: int) -> Trajectory:
        """Generate sample trajectories using given policy.

        Args:
            policy (Policy): Policy distribution over actions given the current history.
            time_horizon (int): Trajectory length.
            n_samples (int): Number of trajectories to simulate.

        Returns:
            Trajectory: Tuple of matricies representing trajectories, each of shape
                (n_samples, time_horizon)
        """
        states = self.states
        actions = self.actions
        history_length = self.history_length
        transition_probability = self.transition_probability
        initial_state_probability = self.initial_state_probability

        # s_t from time=0, ..., time_horizon
        s_t = np.zeros((n_samples, time_horizon + 1), dtype=np.int32)
        # a_t from time=0, ..., time_horizon-1
        a_t = np.zeros((n_samples, time_horizon), dtype=np.int32)
        # r_t from time=1, ..., time_horizon
        r_t = np.zeros((n_samples, time_horizon), dtype=np.float32)

        # Initial states for each trajectory
        s_t[:, 0] = np.random.choice(
            states, size=(n_samples, ), p=initial_state_probability)

        # Generate n_samples trajectories
        for i in range(n_samples):
            # Initialize history to before-time-0 state state_size
            history = deque(
                (self.state_size, ) * history_length, maxlen=history_length)
            for time in range(0, time_horizon):
                # Update history to include the current state
                history.appendleft(s_t[i, time])
                # Sample action according to policy
                a_dist = policy[tuple(history)]
                action = np.random.choice(actions, p=a_dist)
                # Generate reward and next state
                transitions = transition_probability[tuple(history) +
                                                     (action, )]
                # p[:, 0] is distribution over next states
                # p[i, 1] is reward for transitioning to state i
                state = np.random.choice(states, p=transitions[:, 0])
                reward = transitions[state][1]
                # Save current time step
                s_t[i, time + 1] = state
                a_t[i, time] = action
                r_t[i, time] = reward

        return s_t, a_t, r_t

    @staticmethod
    def random_transition_probability(state_size: int, action_size: int,
                                      history_length: int,
                                      mean_reward: float) -> Distribution:
        """Generate random next state/reward distribution tensor.

        Args:
            state_size (int): Cardinality of the finite state space.
            action_size (int): Cardinality of the finite action space.
            history_length (int): Number of states that matter to the environment when determining
                the next state

        Returns:
            transition_probability (Distribution): Next state/reward distribution given the current
                history and action, P(s_{time+1} | h_t, a_t)
                = transition_probability[h_t[0], ..., h_t[history_length-1], a_t, s_{time+1}]
                Represented as a numpy array of shape
                (state_size, ..., state_size, action_size, state_size, 2)
                so that p = transition_probability[tuple(history)+(action,)][i] is an array of shape
                (2, 1) so that
                    * p[0] = probability next state is state i
                    * p[1] = reward associated with transitioning to the next state

        """
        transition_probability = np.random.random(
            size=transitions_shape(state_size, action_size, history_length))

        # Generate tuples to access p=transition_probability[(tuple(history)+(action,)]
        for history_action in history_action_tuples(state_size, action_size,
                                                    history_length):
            # Normalize the next state distribution
            transitions = transition_probability[history_action]
            transitions[:, 0] = softmax(transitions[:, 0])
            # Use positive and negative reward values
            for i in range(state_size):
                if np.random.random() > 0.5:
                    transitions[i, 1] *= -1
                    transitions[i, 1] += mean_reward

        return transition_probability

    @staticmethod
    def random_pi(lmdp) -> Policy:
        """Generate random policy tensor.

        Args:
            lmdp (FLMDP): L-MDP to create the policy for.

        random_policy (Distribution): Policy distribution over actions given the current history.

        """
        state_size = lmdp.state_size
        action_size = lmdp.action_size
        history_length = lmdp.history_length

        random_policy = np.random.random(
            size=policy_shape(state_size, action_size, history_length))

        for history in history_tuples(state_size, history_length):
            # Normalize the next action distribution
            random_policy[history] = softmax(random_policy[history])

        return random_policy

    @staticmethod
    def scips_approximable_pi(lmdp,
                              gamma: float,
                              sigma: float,
                              time_horizon=100,
                              n_samples=1000) -> Policy:
        """Generate policy tensor under SCIPS assumption.

        Args:
            lmdp (FLMDP): FLMDP for which to make SCIPS approximable policy.
            gamma (float): Time discounting parameter used in the SCIPS,
                in [0.0, 1.0].
            sigma (float): Standard deviaton of noise added to policy.
            time_horizon (int): Trajectory length.
            n_samples (int): Number of trajectories to simulate.

        policy (Distribution): Policy distribution over actions given the current history.

        """

        # Start with a random policy
        random_policy = FLMDP.random_pi(lmdp=lmdp)

        # Simuate some trajectories
        s_t, r_t, a_t = lmdp.simulate(
            policy=random_policy,
            time_horizon=time_horizon,
            n_samples=n_samples)

        # Fit the policy to SCIPS
        scips = sparsity_corrected_approx(
            states=s_t, actions=a_t, rewards=r_t, gamma=gamma, lmdp=lmdp)

        # Add noise
        for history_action in scips:
            scips[history_action] += np.random.normal(loc=0, scale=sigma)

        # Normalize the next action distribution
        for history in history_tuples(lmdp.state_size, lmdp.history_length):
            scips[history] = l1_normalize(scips[history])

        return scips
