""" Utility functions """

from typing import Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """ Returns the softmax probabilities of a 1 dimensional vector of reals. """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def l1_normalize(x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """ Returns the l_1 normalized probabilities of a 1 dimensional vector of reals."""
    return x / x.sum()


def policy_shape(state_size: int, action_size: int,
                 history_length: int) -> Tuple:
    """ Returns a tuple with the shape of an array containing policy probabilities. """
    return ((state_size, ) * history_length) + (action_size, )
