import pathlib

from typing import List
from data_util import datasets_path, MDataset
from objects import GPUType
from .graph_meta import Graph


class DatasetFactory:
    @staticmethod
    def _load_data(gpu_type: GPUType) -> List[Graph]:
        data_dir = pathlib.Path(datasets_path) / f"{gpu_type}"
        # Load data from directory
        ...

    @staticmethod
    def create_dataset(gpu_type: GPUType, normalization: str) -> MDataset:
        input_data, target_data = DatasetFactory._load_data(gpu_type)
        dataset = MDataset(input_data, target_data)
        return dataset
