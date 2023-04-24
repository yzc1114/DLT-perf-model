import pathlib
from functools import lru_cache
from itertools import count
from typing import List, Tuple, Union, Callable, Optional, Dict

import torch
from sklearn import preprocessing
from torch.utils.data import random_split, Dataset

from objects import Environment, DatasetType
from util import datasets_path, Normalizer
from .graph import Graph


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


class DatasetFactory:
    @staticmethod
    def _load_graphs(environment: Environment, train_or_val: str = "train", dummy: bool = False) -> List[Graph]:
        if dummy:
            return list(Graph.from_data(None, dummy=True) for _ in range(100))
        data_dir = pathlib.Path(datasets_path) / f"{environment}"
        # Load data from directory
        return list()

    @staticmethod
    def _split_dataset(dataset: MDataset, train=0.7, val=0.3, seed=42) -> Tuple[MDataset, MDataset]:
        # Set local random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Split the dataset into training, validation/test sets
        train, val = random_split(dataset, [train, val], generator=generator)

        return train, val

    @staticmethod
    def create_dataset(environment: Environment,
                       normalization: str,
                       dataset_type: DatasetType = DatasetType.Subgraph,
                       dummy: bool = False,
                       return_graph_mapping: bool = False,
                       **kwargs) -> Union[
        Tuple[MDataset, MDataset],
        Tuple[MDataset, MDataset, Tuple[Dict[int, Graph], Dict[int, Graph]]],
    ]:
        dataset_creator = {
            DatasetType.OP: DatasetFactory._create_op_dataset,
            DatasetType.Grouping: DatasetFactory._create_graph_grouping_dataset,
            DatasetType.Subgraph: DatasetFactory._create_subgraph_dataset,
        }
        creator = dataset_creator[dataset_type]
        train_graphs = DatasetFactory._load_graphs(environment, train_or_val="train", dummy=dummy)
        val_graphs = DatasetFactory._load_graphs(environment, train_or_val="val", dummy=dummy)
        train, train_mapping = creator(train_graphs, normalization, dummy, **kwargs)
        val, val_mapping = creator(val_graphs, normalization, dummy, **kwargs)
        if not return_graph_mapping:
            return train, val
        return train, val, (train_mapping, val_mapping)

    @staticmethod
    @lru_cache(maxsize=None)
    def _create_op_dataset(graphs: List[Graph],
                           normalization: str,
                           **kwargs) -> Tuple[MDataset, Dict[int, Graph]]:
        op_X, op_Y = list(), list()
        data_idx_to_graph = dict()
        counter = iter(count())
        for graph in graphs:
            X, Y = graph.Serial_feature_extractor.node_features()
            op_X.extend(X)
            op_Y.extend(Y)
            for i in range(len(X)):
                data_idx_to_graph[next(counter)] = graph

        dataset = MDataset(op_X, op_Y, normalization)
        return dataset, data_idx_to_graph

    @staticmethod
    @lru_cache(maxsize=None)
    def _create_graph_grouping_dataset(graphs: List[Graph],
                                       normalization: str,
                                       subgraph_count=10,
                                       **kwargs) -> Tuple[MDataset, Dict[int, Graph]]:
        graph_X, graph_Y = list(), list()
        data_idx_to_graph = dict()
        for i, graph in enumerate(graphs):
            x, y = graph.GNN_based_feature_extractor.full_graph_feature(subgraph_count=subgraph_count)
            graph_X.append(x)
            # graph_Y.append(label.subgraph_durations)
            graph_Y.append(y)
            data_idx_to_graph[i] = graph
        dataset = MDataset(graph_X, graph_Y, normalization)
        return dataset, data_idx_to_graph

    @staticmethod
    @lru_cache(maxsize=None)
    def _create_subgraph_dataset(graphs: List[Graph],
                                 normalization: str,
                                 subgraph_node_size=10,
                                 **kwargs) -> Tuple[MDataset, Dict[int, Graph]]:
        subgraph_X, subgraph_Y = list(), list()
        data_idx_to_graph = dict()
        counter = iter(count())
        for graph in graphs:
            X, Y = graph.Serial_feature_extractor.subgraph_features(subgraph_node_size=subgraph_node_size)
            subgraph_X.extend(X)
            # Y = list()
            # for label in labels:
            #     Y.append(label.subgraph_durations)
            #     Y.append(label.node_durations)
            subgraph_Y.extend(Y)
            for i in range(len(X)):
                data_idx_to_graph[next(counter)] = graph
        dataset = MDataset(subgraph_X, subgraph_Y, normalization)
        return dataset, data_idx_to_graph
