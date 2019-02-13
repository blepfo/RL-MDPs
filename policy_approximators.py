""" Approximates an RL policy from sample episodes. """

import itertools as it
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from _types import Action, History, Position, State, Policy
from utils import policy_shape


def naive_approx(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                 history_length: int) -> Policy:
    """Approximates a policy as a Monte-Carlo estimate of the action taken from each history."""
    time_horizon = actions.shape[1]
    n_samples = actions.shape[0]

    assert rewards.shape == actions.shape
    assert states.shape[0] == n_samples
    assert states.shape[1] - 1 == time_horizon

    state_size: int = max(states)
    action_size: int = max(actions)

    history_counts: Dict[History, int] = {}
    history_action_counts: Dict[Tuple[History, Action], int] = {}
    history_deque: Deque[State] = deque(
        (0, ) * history_length, maxlen=history_length)

    for i in range(n_samples):
        trajectory = states[i, :]
        for time, state in it.islice(
                enumerate(trajectory), 0, time_horizon - 1):
            history_deque.appendleft(state)
            history = np.asarray(history_deque)
            history_counts[history] = history_counts.get(history, 0) + 1
            context = (history, actions[i, time])
            history_action_counts[context] = history_action_counts.get(
                context, 0) + 1

    naive_probabilities: Policy = np.ndarray(
        shape=policy_shape(state_size, action_size, history_length))

    for (history, action), count in history_action_counts.items():
        naive_probabilities[history, action] = float(count) / float(
            history_counts[history])

    return naive_probabilities


def sc_probability(history: History, action: Action, gamma: float,
                   positional_state_counts: dict,
                   positional_state_action_counts: dict,
                   history_length: int) -> float:
    """Gets the probability estimate of an action given a specific history.

    """
    probability = 0

    for tau in range(history_length):
        if positional_state_counts.get((history[tau], tau), 0.0) > 0.0:
            probability += (gamma**tau) * \
                positional_state_action_counts.get((history[tau], tau, action), 0.0) \
                / positional_state_counts.get((history[tau], tau), 0.0)

    return probability * (1.0 - gamma) / (1.0 - gamma**(history_length))


def sparsity_corrected_approx(states: np.ndarray, actions: np.ndarray,
                              rewards: np.ndarray, gamma: float,
                              lmdp) -> Policy:
    """ Approximates a policy using the sparsity corrected method.

    """
    time_horizon = actions.shape[1]
    n_samples = actions.shape[0]

    history_length = lmdp.history_length
    state_size = lmdp.state_size
    action_size = lmdp.action_size

    assert rewards.shape == actions.shape
    assert states.shape[0] == n_samples
    assert states.shape[1] - 1 == time_horizon

    positional_state_counts: Dict[Tuple[State, Position], int] = {}
    positional_state_action_counts: Dict[Tuple[State, Position, Action],
                                         int] = {}
    history_deque: Deque[State] = deque(
        (0, ) * history_length, maxlen=history_length)

    for i in range(n_samples):
        trajectory = states[i, :]
        for time, state in it.islice(
                enumerate(trajectory), 0, time_horizon - 1):
            history_deque.appendleft(state)
            for k, state_in_history in enumerate(history_deque):
                state_context: Tuple[State, Position] = (state_in_history, k)
                positional_state_counts[
                    state_context] = positional_state_counts.get(
                        state_context, 0) + 1
                state_action_context: Tuple[State, Position, Action] = (
                    state_in_history, k, actions[i, time])
                positional_state_action_counts[state_action_context] = \
                    positional_state_action_counts.get(
                        state_action_context, 0) + 1

    history_sizes = it.repeat(list(range(state_size + 1)), history_length)
    history_action_sizes = it.chain(history_sizes, [list(range(action_size))])
    history_actions = it.product(*history_action_sizes)

    sc_history_action_probabilities = np.ndarray(
        shape=policy_shape(state_size, action_size, history_length))
    for history_action in history_actions:
        history = np.asarray(history_action[:-1])
        action = history_action[-1]
        sc_history_action_probabilities[history_action] = \
            sc_probability(history, action, gamma, positional_state_counts,
                           positional_state_action_counts, history_length)

    return sc_history_action_probabilities
