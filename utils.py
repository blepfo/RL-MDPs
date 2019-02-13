""" Utility functions """

from typing import Tuple
import itertools as it

import numpy as np


def l1_normalize(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """ Returns the l_1 normalized probabilities of a 1 dimensional vector of reals."""
    return x / x.sum()


def policy_shape(state_size: int, action_size: int,
                 history_length: int) -> Tuple:
    """ Returns a tuple with the shape of an array containing policy probabilities.

    Note that states go from 0 to state_size because state_size is the before-time padding state.
    """
    return ((state_size, ) * history_length) + (action_size, )


def transitions_shape(state_size: int, action_size: int, history_length: int):
    """ Returns a tuple with the shape of the transition probabilities and rewards array."""
    return ((state_size + 1, ) * history_length) + (action_size, state_size, 2)


def history_tuples(state_size: int, history_length: int):
    """ Generate tuples of states of history_length

    Note that states go from 0 to state_size because state_size is the before-time padding state.
    """
    history_sizes = it.repeat(list(range(state_size + 1)), history_length)
    return it.product(*history_sizes)


def history_action_tuples(state_size: int, action_size: int,
                          history_length: int):
    """ Generate tuplies of states in the history and actions.

    Note that states go from 0 to state_size because state_size is the before-time padding state.
    """
    sizes = list(it.repeat(list(range(state_size + 1)),
                           history_length)) + [list(range(action_size))]
    return it.product(*sizes)
