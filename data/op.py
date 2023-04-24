from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List

import numpy as np


class OperatorType(Enum):
    Add = 0
    MatMul = 1
    Conv1d = 2
    Conv2d = 3
    MaxPooling2d = 4
    AvgPooling2d = 5
    Relu = 6
    Sigmoid = 7
    Softmax = 8
    Tanh = 9

    @lru_cache(maxsize=None)
    def encode(self, method="one-hot") -> List:
        op_types = [op for op in OperatorType]
        if method == "one-hot":
            return [1 if self == op_type_ else 0 for op_type_ in op_types]
        elif method == "frequency":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Invalid method. Must be 'one-hot' or 'frequency'.")


class Operator:
    def __init__(self,
                 operator_type: OperatorType,
                 input_tensor_size: int,
                 weight_tensor_size: int,
                 output_tensor_size: int,
                 FLOPS: float,
                 hyper_parameters: Optional[Tuple[Union[float, int]]]
                 ):
        self.operator_type: OperatorType = operator_type
        self.input_tensor_size: int = input_tensor_size
        self.weight_tensor_size: int = weight_tensor_size
        self.output_tensor_size: int = output_tensor_size
        self.FLOPS: float = FLOPS
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    def to_feature(self, op_type_encoding="one-hot", mode="complex"):
        if mode == "complex":
            complex_feature_vector = [
                *self.operator_type.encode(method=op_type_encoding),
                self.input_tensor_size,
                self.weight_tensor_size,
                self.output_tensor_size,
                self.FLOPS,
                *self.hyper_parameters
            ]
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                *self.operator_type.encode(method=op_type_encoding),
                self.input_tensor_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")
