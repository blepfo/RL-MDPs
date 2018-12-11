""" Computes the stepwise importance sampling estimate of return. """

from typing import List
from collections import deque


def step_is(pi_b: dict, pi_e: dict, state_samples: List[List[int]], action_samples: List[List[int]],
            reward_samples: List[List[float]], l: int, gamma: float) -> float:
    """ Estimates the return of pi_e from pi_b and some episodes from pi_b. """
    cumulative_reward: float = 0.0
    for time, (state_sample, action_sample, reward_sample) in \
            enumerate(zip(state_samples, action_samples, reward_samples)):
        weighted_return: float = 0.0
        likelyhood_ratio: float = 1.0
        history = deque((0,) * l, maxlen=l)
        for state, action, reward in zip(state_sample, action_sample, reward_sample):
            history.appendleft(state)
            likelyhood_ratio *= pi_e[(history, action)] / \
                pi_b[(history, action)]
            weighted_return += likelyhood_ratio * reward * gamma ** time
        cumulative_reward += weighted_return
    return cumulative_reward
