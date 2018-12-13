""" Approximates an RL policy from sample episodes. """

from collections import deque
from itertools import islice, product
from typing import Deque, Dict, Tuple

import pandas as pd
from numpy import unique

# actions_path = "./output/test_actions.csv"
# actions = pd.read_csv(actions_path, header=None)
# states_path = "./output/test_states.csv"
# states = pd.read_csv(states_path, header=None)
# rewards_path = "./output/test_rewards.csv"
# rewards = pd.read_csv(rewards_path, header=None)

State = int
Action = int
Reward = float
Position = int
History = Tuple[State, ...]


def naive_approx(actions: pd.DataFrame, states: pd.DataFrame, rewards: pd.DataFrame, l: int) \
        -> dict:
    """ Approximates a policy as a Monte-Carlo estimate of the action taken from each history. """
    T = actions.shape[1]
    m = actions.shape[0]
    assert rewards.shape == actions.shape
    assert states.shape[0] == m
    assert states.shape[1] - 1 == T

    history_counts: Dict[History, int] = {}
    history_action_counts: Dict[Tuple[History, Action], int] = {}
    history_deque: Deque[State] = deque((0,) * l, maxlen=l)
    for i, trajectory in states.iterrows():
        for j, state in islice(trajectory.items(), 0, T):
            history_deque.appendleft(state)
            history_counts[tuple(history_deque)] = history_counts.get(
                tuple(history_deque), 0) + 1
            context = (tuple(history_deque), actions[j][i])
            history_action_counts[context] = history_action_counts.get(
                context, 0) + 1

    naive_probabilities: Dict[Tuple[History, Action], float] = {}
    for (history, action), count in history_action_counts.items():
        naive_probabilities[(history, action)] = float(
            count) / float(history_counts[history])
    return naive_probabilities


def sc_probability(history: Tuple[State, ...], action: int, Gamma: float,
                   positional_state_counts: dict, positional_state_action_counts: dict, l: int) \
        -> float:
    """ Gets the probability estimate of an action given a specific history. """
    probability = 0
    for i in range(l):
        if positional_state_counts.get((history[i], i), 0.0) > 0.0:
            probability += Gamma**i * \
                positional_state_action_counts.get((history[i], i, action), 0.0) \
                / positional_state_counts.get((history[i], i), 0.0)
    return probability * (1.0 - Gamma) / (1.0 - Gamma**(l))


def sparsity_corrected_approx(actions: pd.DataFrame, states: pd.DataFrame, rewards: pd.DataFrame,
                              l: int, Gamma: float) -> dict:
    """ Approximates a policy using the sparsity corrected method. """

    T = actions.shape[1]
    m = actions.shape[0]
    assert rewards.shape == actions.shape
    assert states.shape[0] == m
    assert states.shape[1] - 1 == T

    positional_state_counts: Dict[Tuple[State, Position], int] = {}

    positional_state_action_counts: Dict[Tuple[State, Position, Action], int] = {
    }
    history_deque: Deque[State] = deque((0,) * l, maxlen=l)
    for i, trajectory in states.iterrows():
        for j, state in islice(trajectory.items(), 0, T):
            history_deque.appendleft(state)
            for k, state_in_history in enumerate(history_deque):
                state_context: Tuple[State, Position] = (state_in_history, k)
                positional_state_counts[state_context] = positional_state_counts.get(
                    state_context, 0) + 1
                state_action_context: Tuple[State, Position, Action] = (
                    state_in_history, k, actions[j][i])
                positional_state_action_counts[state_action_context] = \
                    positional_state_action_counts.get(
                        state_action_context, 0) + 1

    sc_history_action_probabilities = {}
    for history, action in \
            product(product(list(unique(states.values)) + [0], repeat=l), unique(actions.values)):
        sc_history_action_probabilities[history, action] = \
            sc_probability(history, action, Gamma, positional_state_counts,
                           positional_state_action_counts, l)
    return sc_history_action_probabilities