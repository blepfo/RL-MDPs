""" Computes the stepwise importance sampling estimate of return. """

from collections import deque
import numpy as np

from _types import Policy


def step_is(pi_b: Policy, pi_e: Policy, state_samples: np.ndarray,
            action_samples: np.ndarray, reward_samples: np.ndarray,
            history_length: int, gamma: float) -> float:
    """ Estimates the return of pi_e from pi_b and some episodes from pi_b.

    Args:
        TODO

    Returns:
        cumulative_reward (float): TODO

    """
    cumulative_reward = 0.0

    for state_sample, action_sample, reward_sample in \
            zip(state_samples, action_samples, reward_samples):
        # Compute weighted return for each trajectory
        weighted_return = 0.0
        likelyhood_ratio = 1.0

        history = deque((0, ) * history_length, maxlen=history_length)

        for time, (state, action, reward) in \
                enumerate(zip(state_sample, action_sample, reward_sample)):
            history.appendleft(state)

            # Update likelihood ratio
            numerator = pi_e[tuple(history) + (action, )]

            if isinstance(pi_b, dict):
                # Approximators represent policy as dict
                denominator = pi_b.get(tuple(history) + (action, ), 0)
                assert False  # This should never happen.
            else:
                # FLMDP simulation represents policy as np.ndarray
                denominator = pi_b[tuple(history) + (action, )]

            if denominator == 0:
                if numerator == 0:
                    # 0 / 0
                    continue
                else:
                    # Nonzero numerator  / 0
                    continue

            likelyhood_ratio *= (numerator / denominator)

            # Update current trajectory return
            weighted_return += likelyhood_ratio * reward * (gamma**time)

        cumulative_reward += weighted_return

    return cumulative_reward / reward_samples.shape[0]
