from enum import Enum
from functools import lru_cache
from typing import List

from torch.optim import Adam, RMSprop, SGD


class ModelType(Enum):
    GBDT = 0
    GNN = 1
    MLP = 2
    PerfNet = 3
    RNN = 4
    Transformer = 5


class Environment:
    def __init__(self, gpu_type: 'GPUType', framework: str, cuda_version: str):
        self.gpu_type: GPUType = gpu_type
        self.framework: str = framework
        self.cuda_version: str = cuda_version

    def __str__(self):
        return f"{self.gpu_type.name}_{self.framework}_{self.cuda_version}"

    def __repr__(self):
        return self.__str__()


class GPUType(Enum):
    RTX2080Ti = 0
    T4 = 1
    RTX4090 = 2
    P40 = 3
    K80 = 4


class DatasetType(Enum):
    OP = 0
    Grouping = 1
    Subgraph = 2


class OptimizerType(Enum):
    Adam = Adam
    SGD = SGD
    RMSProp = RMSprop

    @lru_cache(maxsize=None)
    def encode(self, method="one-hot") -> List:
        om_types = [om for om in OptimizerType]
        if method == "one-hot":
            return [1 if self == om_type_ else 0 for om_type_ in om_types]
        else:
            raise ValueError(
                "Invalid method. Must be 'one-hot'.")
