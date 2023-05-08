import pathlib
from typing import List, Union, Dict

import numpy as np

datasets_path = str(pathlib.Path(__file__) / "datasets")


class Normalizer:
    DefaultKey = "default"

    def __init__(self, x_scaler_dict, y_scaler_dict):
        self.x_scaler_dict = x_scaler_dict
        self.y_scaler_dict = y_scaler_dict

    def __call__(self, x: Union[Dict, np.ndarray] | None=None, y: Union[Dict, np.ndarray] | None=None, inversion: bool = False):
        result = list()

        def call_scaler(scaler, value):
            if scaler is None:
                return value
            if isinstance(value, float):
                value = np.array([value])
            if isinstance(value, list) or isinstance(value, tuple):
                value = np.array(value)
            transform_func = scaler.transform if not inversion else scaler.inverse_transform
            if len(value.shape) == 1:
                value = value.reshape(1, -1)
                return transform_func(value)[0]
            else:
                return transform_func(value)

        def n(data, scaler_dict):
            if data is None:
                return
            if isinstance(data, dict):
                normed = dict()
                for k, v in data.items():
                    scaler = scaler_dict[k]
                    v_ = call_scaler(scaler, v)
                    normed[k] = v_
                result.append(normed)
            elif isinstance(data, np.ndarray):
                scaler = scaler_dict[Normalizer.DefaultKey]
                result.append(call_scaler(scaler, data))
            else:
                raise ValueError("Invalid data type. Data must be a dict or a np.ndarray.")

        n(x, self.x_scaler_dict)
        n(y, self.y_scaler_dict)

        result = tuple(filter(lambda r: r is not None, result))
        if len(result) == 1:
            return result[0]
        return result

    # @staticmethod
    # def from_data(X: Union[List[Dict], List[np.ndarray]], Y: Union[List[Dict], List[np.ndarray]],
    #               scaler_class=None) -> 'Normalizer':
    #     def get_scaler_dict(data):
    #         scaler_dict = dict()
    #         if isinstance(data[0], dict):
    #             keys = data[0].keys()
    #             for k in keys:
    #                 if not k.endswith(Normalizer.NormKeySuffix):
    #                     scaler_dict[k] = lambda x: x
    #                     continue
    #                 scaler = scaler_class()
    #                 d = [d_[k] for d_ in data]
    #                 scaler.fit(d)
    #                 scaler_dict[k] = scaler
    #             return scaler_dict
    #         elif isinstance(data[0], np.ndarray):
    #             scaler = scaler_class()
    #             scaler.fit(data)
    #             scaler_dict[Normalizer.DefaultKey] = scaler
    #         else:
    #             raise ValueError("Invalid data type. Data must be a dict or a np.ndarray.")
    #     x_scaler_dict = get_scaler_dict(X)
    #     y_scaler_dict = get_scaler_dict(Y)
    #     return Normalizer(x_scaler_dict, y_scaler_dict)


def pad_np_vectors(v: List[np.ndarray] | np.ndarray, max_size=None):
    if max_size is None:
        if isinstance(v, np.ndarray):
            raise ValueError("maxsize must be specified is v is np.ndarray")
        if isinstance(v, list):
            max_size = np.amax([f.size for f in v])

    nv = list()
    for l in v:
        if l.size < max_size:
            nv.append(
                np.pad(l, (0, max_size - l.size))
            )
        else:
            nv.append(l)
    return nv
