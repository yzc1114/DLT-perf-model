import pathlib
from typing import List

import numpy as np
from sklearn import preprocessing

datasets_path = str(pathlib.Path(__file__) / "datasets")


class Normalizer:
    def __init__(self, x_scaler, y_scaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def _normalize(self, x=None, y=None):
        def n(v, scaler: preprocessing.MinMaxScaler):
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

    @staticmethod
    def from_data(X, Y, scaler_class=None) -> 'Normalizer':
        x_scaler = scaler_class()
        y_scaler = scaler_class()
        x_scaler.fit(X)
        y_scaler.fit(Y)
        return Normalizer(x_scaler, y_scaler)


def pad_np_vectors(v: List[np.array]):
    max_size = np.max(f.size for f in v)
    nv = list()
    for l in v:
        if l.size < max_size:
            nv.append(
                np.pad(l, (0, max_size - l.size))
            )
    return nv
