import pathlib
from typing import Callable, Optional, Union, List

import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset

datasets_path = str(pathlib.Path(__file__) / "datasets")


class MDataset(Dataset):
    def __init__(self, data, targets, normalization: Union[str, Callable] = "standard"):
        self.data = data
        self.targets = targets
        self.normalizer: Optional[Normalizer] = None
        if isinstance(normalization, str):
            self._init_normalizer(normalization)
        elif callable(normalization):
            self.normalizer = normalization
        else:
            raise ValueError("Invalid normalization. Normalization must be a string or a callable object.")

    def _init_normalizer(self, normalization: str):
        if normalization == "Standard":
            self.normalizer = Normalizer.from_data(
                X=self.data, Y=self.targets, scaler_class=preprocessing.StandardScaler),
        elif normalization == "MinMax":
            self.normalizer = Normalizer.from_data(
                X=self.data, Y=self.targets, scaler_class=preprocessing.MinMaxScaler),
        else:
            raise ValueError("Invalid normalization. string normalization must be 'standard'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.normalizer:
            x, y = self.normalizer(x, y)

        return x, y

    def get_normalizer(self) -> Callable:
        return self.normalizer


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
