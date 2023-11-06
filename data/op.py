from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict

import numpy as np


class OperatorType(Enum):
    Dummy = 0
    Add = 1
    MatMul = 2
    Conv1d = 3
    Conv2d = 4
    MaxPooling2d = 5
    AvgPooling2d = 6
    Relu = 7
    Sigmoid = 8
    Softmax = 9
    Tanh = 10

    @lru_cache(maxsize=None)
    def encode(self, method) -> List:
        op_types = [op for op in OperatorType]
        if method == "one-hot":
            return [1 if self == op_type_ else 0 for op_type_ in op_types]
        else:
            raise ValueError(
                "Invalid method. Must be in ['one-hot'].")


class OperatorMode(Enum):
    Forward = 0
    Backward = 1
    Update = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_modes = [mode for mode in OperatorMode]
        return [1 if self == op_mode_ else 0 for op_mode_ in op_modes]


class Operator:
    def __init__(self,
                 operator_type: OperatorType,
                 operator_mode: OperatorMode,
                 input_tensor_size: int=0,
                 weight_tensor_size: int=0,
                 output_tensor_size: int=0,
                 FLOPS: float=0,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type: OperatorType = operator_type
        self.operator_mode: OperatorMode = operator_mode
        self.input_tensor_size: int = input_tensor_size
        self.weight_tensor_size: int = weight_tensor_size
        self.output_tensor_size: int = output_tensor_size
        self.FLOPS: float = FLOPS
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    @staticmethod
    def dummy_op():
        return Operator(OperatorType.Dummy, OperatorMode.Forward, 0, 0, 0, 0)

    def to_feature_array(self, op_type_encoding, mode):
        if mode == "complex":
            complex_feature_vector = [
                *self.operator_type.encode(method=op_type_encoding),
                *self.operator_mode.encode(),
                self.input_tensor_size,
                self.weight_tensor_size,
                self.output_tensor_size,
                self.FLOPS,
            ]
            if self.hyper_parameters is not None:
                complex_feature_vector.extend(self.hyper_parameters)
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                *self.operator_type.encode(method=op_type_encoding),
                *self.operator_mode.encode(),
                self.input_tensor_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")
