from typing import TypeVar

import numpy as np

# Define alias types for clearer documentation
State = TypeVar(int)
Action = TypeVar(int)
Reward = TypeVar(float)

History = TypeVar(np.array)

Distribution = TypeVar(np.ndarray)

Position = TypeVar(int)
