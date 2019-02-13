""" Defines alias types for documenation. """

from typing import Type, Tuple

import numpy as np

State = int
Action = int
Reward = float

Trajectory = Tuple[np.ndarray, np.ndarray, np.ndarray]
History = Type[np.ndarray]

Distribution = Type[np.ndarray]

Policy = Type[np.ndarray]

Position = int
