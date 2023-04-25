import pathlib
from typing import List, Union, Dict

import numpy as np
from sklearn import preprocessing

datasets_path = str(pathlib.Path(__file__) / "datasets")


class Normalizer:
    DefaultKey = "default"

    def __init__(self, x_scaler_dict, y_scaler_dict):
        self.x_scaler_dict = x_scaler_dict
        self.y_scaler_dict = y_scaler_dict

    class _Impl:
        def __init__(self, x_scaler, y_scaler):
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler

        def _normalize(self, x=None, y=None):
            def n(v, scaler):
                if v is None:
                    return None
                return scaler.transform(x)

            return n(x, self.x_scaler), n(y, self.y_scaler)

        def __call__(self, x, y):
            result = self._normalize(x, y)
            result = tuple(filter(lambda r: r is not None, result))
            if len(result) == 1:
                return result[0]
            return result

    def __call__(self, x: Union[Dict, np.ndarray], y: Union[Dict, np.ndarray]):
        result = list()

        def n(data, scaler_dict):
            if isinstance(data, dict):
                normed = dict()
                for k, v in data.items():
                    normed[k] = scaler_dict[k](v)
                result.append(normed)
            elif isinstance(data, np.ndarray):
                result.append(scaler_dict[Normalizer.DefaultKey](data))
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


def pad_np_vectors(v: List[np.array]):
    max_size = np.max(f.size for f in v)
    nv = list()
    for l in v:
        if l.size < max_size:
            nv.append(
                np.pad(l, (0, max_size - l.size))
            )
    return nv
