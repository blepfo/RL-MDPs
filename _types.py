import numpy as np
from typing import TypeVar

# Define alias types for clearer documentation
State = TypeVar(int)
Action = TypeVar(int)
Reward = TypeVar(float)

History = TypeVar(np.array)

Distribution = TypeVar(np.ndarray)
