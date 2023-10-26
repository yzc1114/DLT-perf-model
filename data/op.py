from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict

import numpy as np

# op_freq: Dict['OperatorType', float] | None = None


# def set_operator_freq(op_freq_count_dict: Dict['OperatorType', int]):
#     global op_freq
#     op_freq = dict()
#     total_count = sum(op_freq_count_dict.values())
#     for op, count in op_freq_count_dict.items():
#         op_freq[op] = 1. * count / total_count


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


class Operator:
    def __init__(self,
                 operator_type: OperatorType,
                 input_tensor_size: int,
                 weight_tensor_size: int,
                 output_tensor_size: int,
                 FLOPS: float,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type: OperatorType = operator_type
        self.input_tensor_size: int = input_tensor_size
        self.weight_tensor_size: int = weight_tensor_size
        self.output_tensor_size: int = output_tensor_size
        self.FLOPS: float = FLOPS
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    @staticmethod
    def dummy_op():
        return Operator(OperatorType.Dummy, 0, 0, 0, 0)

    def to_feature_array(self, op_type_encoding, mode):
        if mode == "complex":
            complex_feature_vector = [
                *self.operator_type.encode(method=op_type_encoding),
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
                self.input_tensor_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")
