from enum import Enum
from functools import lru_cache
from typing import List


class GPUType(Enum):
    RTX2080Ti = 0
    T4 = 1
    RTX4090 = 2
    P40 = 3
    K80 = 4


class OptimizerType(Enum):
    Adam = 0
    Momentum = 1
    SGD = 2
    RMSProp = 3

    @lru_cache(maxsize=None)
    def encode(self, method="one-hot") -> List:
        om_types = [om for om in OptimizerType]
        if method == "one-hot":
            return [1 if self == om_type_ else 0 for om_type_ in om_types]
        else:
            raise ValueError(
                "Invalid method. Must be 'one-hot'.")