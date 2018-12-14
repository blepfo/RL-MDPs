""" Computes the stepwise importance sampling estimate of return. """

from typing import List
from collections import deque


def step_is(pi_b: dict,
            pi_e: dict,
            state_samples: List[List[int]],
            action_samples: List[List[int]],
            reward_samples: List[List[float]],
            l: int,
            gamma: float) -> float:
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

        history = deque((0,) * l, maxlen=l)

        for time, (state, action, reward) in \
                enumerate(zip(state_sample, action_sample, reward_sample)):
            history.appendleft(state)

            # Update likelihood ratio
            numerator = pi_e[tuple(history)+(action,)]

            if isinstance(pi_b, dict):
                # Approximators represent pi as dict
                denominator = pi_b.get(tuple(history)+(action,), 0)
            else:
                # FLMDP simulation represents pi as np.ndarray
                denominator = pi_b[tuple(history)+(action,)]

            if denominator == 0:
                if numerator == 0:
                    # 0 / 0
                    continue
                else:
                    # Nonzero numerator  / 0
                    continue

            likelyhood_ratio *= (numerator / denominator)

            # Update current trajectory return
            weighted_return += likelyhood_ratio * reward * (gamma ** time)

        cumulative_reward += weighted_return

    return cumulative_reward
