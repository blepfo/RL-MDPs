""" Approximates an RL policy from sample episodes. """

import itertools as it
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from _types import Action, History, Position, State

# actions_path = "./output/test_actions.csv"
# actions = pd.read_csv(actions_path, header=None)
# states_path = "./output/test_states.csv"
# states = pd.read_csv(states_path, header=None)
# rewards_path = "./output/test_rewards.csv"
# rewards = pd.read_csv(rewards_path, header=None)


def naive_approx(states: np.ndarray,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 l: int) -> dict:
    """Approximates a policy as a Monte-Carlo estimate of the action taken from each history.

    """
    T = actions.shape[1]
    m = actions.shape[0]

    assert rewards.shape == actions.shape
    assert states.shape[0] == m
    assert states.shape[1] - 1 == T

    history_counts: Dict[History, int] = {}
    history_action_counts: Dict[Tuple[History, Action], int] = {}
    history_deque: Deque[State] = deque((0,) * l, maxlen=l)

    for i in range(m):
        trajectory = states[i, :]
        for t, state in it.islice(enumerate(trajectory), 0, T-1):
            history_deque.appendleft(state)
            history_counts[tuple(history_deque)] = history_counts.get(
                tuple(history_deque), 0) + 1
            context = (tuple(history_deque), actions[i, t])
            history_action_counts[context] = history_action_counts.get(
                context, 0) + 1

    naive_probabilities: Dict[tuple, float] = {}

    for (history, action), count in history_action_counts.items():
        naive_probabilities[tuple(history) + (action,)] = float(
            count) / float(history_counts[history])

    return naive_probabilities


def sc_probability(history: History,
                   action: Action,
                   Gamma: float,
                   positional_state_counts: dict,
                   positional_state_action_counts: dict,
                   l: int) -> float:
    """Gets the probability estimate of an action given a specific history.

    """
    probability = 0

    for tau in range(l):
        if positional_state_counts.get((history[tau], tau), 0.0) > 0.0:
            probability += (Gamma**tau) * \
                positional_state_action_counts.get((history[tau], tau, action), 0.0) \
                / positional_state_counts.get((history[tau], tau), 0.0)

    return probability * (1.0 - Gamma) / (1.0 - Gamma**(l))


def sparsity_corrected_approx(states: np.ndarray,
                              actions: np.ndarray,
                              rewards: np.ndarray,
                              Gamma: float,
                              lmdp) -> dict:
    """ Approximates a policy using the sparsity corrected method. 

    """
    T = actions.shape[1]
    m = actions.shape[0]

    l = lmdp.l
    mag_S = lmdp.mag_S
    mag_A = lmdp.mag_A

    assert rewards.shape == actions.shape
    assert states.shape[0] == m
    assert states.shape[1] - 1 == T

    positional_state_counts: Dict[Tuple[State, Position], int] = {}
    positional_state_action_counts: Dict[Tuple[State, Position, Action], int] = {
    }
    history_deque: Deque[State] = deque((0,) * l, maxlen=l)

    for i in range(m):
        trajectory = states[i, :]
        for t, state in it.islice(enumerate(trajectory), 0, T-1):
            history_deque.appendleft(state)
            for k, state_in_history in enumerate(history_deque):
                state_context: Tuple[State, Position] = (state_in_history, k)
                positional_state_counts[state_context] = positional_state_counts.get(
                    state_context, 0) + 1
                state_action_context: Tuple[State, Position, Action] = (
                    state_in_history, k, actions[i, t])
                positional_state_action_counts[state_action_context] = \
                    positional_state_action_counts.get(
                        state_action_context, 0) + 1

    history_sizes = it.repeat(list(range(mag_S+1)), l)
    history_action_sizes = it.chain(history_sizes, [list(range(mag_A))])
    history_actions = it.product(*history_action_sizes)

    sc_history_action_probabilities = {}
    for history_action in history_actions:
        history = history_action[:-1]
        action = history_action[-1]
        sc_history_action_probabilities[history_action] = \
            sc_probability(history, action, Gamma, positional_state_counts,
                           positional_state_action_counts, l)

    return sc_history_action_probabilities
