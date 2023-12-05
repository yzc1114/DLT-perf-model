from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict

import numpy as np

from data.util import get_op_frequency


class OperatorMode(Enum):
    Forward = 0
    Backward = 1
    Update = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_modes = [mode for mode in OperatorMode]
        return [1 if self == op_mode_ else 0 for op_mode_ in op_modes]


class OperatorType(Enum):
    add = 0  # 12
    mul = 1  # 16
    conv2d = 2  # 203
    floordiv = 3  # 23
    sigmoid = 4  # 156
    batchnorm = 5  # 205
    relu = 6  # 152
    iadd = 7  # 35
    dropout = 7  # 212
    silu = 8  # 159
    linear = 9  # 201
    bernoulli = 10  # 234
    adaptive_avg_pool2d = 11  # 199
    layer_norm = 12  # 0
    normalize = 13  # 230
    sub = 14  # 21
    matmul = 15  # 207
    gelu = 16  # 166
    cat = 17  # 213
    clone = 18  # 239
    index = 19  # 233
    softmax = 20  # 224
    truediv = 21  # 25
    matmul_ = 22  # 206
    reshape = 23  # 214
    cross_entrpy = 24  # 181
    # 出现频率过低， 视为一类
    others = 25

    # pad = 25 # 229
    # clamp = 26 # 93
    # avg_pool2d = 27 # 189
    # hardswish = 28 # 243
    # roll = 29 # 231
    # max_pool3d = 30 # 183
    # hardtanh = 31 # 154
    # hardsigmoid = 32 # 244
    # newzeros = 33 # 232
    # mean = 34 # 210
    @lru_cache(maxsize=None)
    def encode(self) -> List:
        ops = [op for op in OperatorType]
        return [1 if self == op else 0 for op in ops]

    @staticmethod
    def get_encode(op_id: int) -> List:
        op_map = {
            12: OperatorType.add,
            16: OperatorType.mul,
            203: OperatorType.conv2d,
            23: OperatorType.floordiv,
            156: OperatorType.sigmoid,
            205: OperatorType.batchnorm,
            152: OperatorType.relu,
            35: OperatorType.iadd,
            212: OperatorType.dropout,
            159: OperatorType.silu,
            201: OperatorType.linear,
            234: OperatorType.bernoulli,
            199: OperatorType.adaptive_avg_pool2d,
            0: OperatorType.layer_norm,
            230: OperatorType.normalize,
            21: OperatorType.sub,
            207: OperatorType.matmul,
            166: OperatorType.gelu,
            213: OperatorType.cat,
            239: OperatorType.clone,
            233: OperatorType.index,
            224: OperatorType.softmax,
            25: OperatorType.truediv,
            206: OperatorType.matmul_,
            214: OperatorType.reshape,
            181: OperatorType.cross_entrpy,
        }
        if op_id in op_map:
            return op_map[op_id].encode()
        else:
            return OperatorType.others.encode()


class OperatorDtype(Enum):
    Float = 0
    Half = 1
    Double = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_dtypes = [dtype for dtype in OperatorDtype]
        return [1 if self == op_dtype_ else 0 for op_dtype_ in op_dtypes]


class Operator:
    def __init__(self,
                 operator_type_id: int,
                 operator_mode: OperatorMode,
                 operator_dtype: OperatorDtype = OperatorDtype.Float,
                 h: int = 0,
                 batch_size: int = 0,
                 FLOPS: int = 0,
                 bytes: int = 0,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type_id: int = operator_type_id
        self.operator_mode: OperatorMode = operator_mode
        self.batch_size: int = batch_size
        self.h: int = h
        self.dtype: OperatorDtype = operator_dtype

        self.FLOPS: int = FLOPS
        self.bytes: int = bytes
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    @staticmethod
    def dummy_op():
        return Operator(0, OperatorMode.Forward)

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id(i: int) -> List:
        l = [0] * 238
        l[i - 1] = 1
        return l

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_static(i: int) -> List:
        return [i]

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_frequency(i: int) -> List:
        pass

    def to_feature_array(self, mode):
        if mode == "complex":
            complex_feature_vector = [
                # *Operator.encode_op_type_id_static(self.operator_type_id),
                *OperatorType.get_encode(self.operator_type_id),
                *self.operator_mode.encode(),
                *self.dtype.encode(),
                self.batch_size,
                self.h,
                self.FLOPS,
                self.bytes,
            ]
            if self.hyper_parameters is not None:
                complex_feature_vector.extend(self.hyper_parameters)
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                # *self.encode_op_type_id_static(self.operator_type_id),
                *OperatorType.get_encode(self.operator_type_id),
                *self.operator_mode.encode(),
                *self.dtype.encode(),
                self.h,
                self.batch_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")
