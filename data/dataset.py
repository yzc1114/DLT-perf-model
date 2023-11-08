import pathlib
import os
import pandas as pds
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict

from torch.utils.data import Dataset

from objects import Environment
from .graph import Graph

datasets_path = str(pathlib.Path(os.getcwd()) / "datasets")


class MDataset(Dataset):
    def __init__(self, features: List[Dict], labels: List[Dict]):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

        return x, y


@lru_cache(maxsize=None)
def load_graphs(environment: Environment, train_or_eval: str = "train", use_dummy: bool = False) -> List[Graph]:
    def _load_graphs():
        if use_dummy:
            return list(Graph.from_data(environment, dummy=True, seed=seed) for seed in range(500))
        data_dir = pathlib.Path(datasets_path) / f"{environment}" / train_or_eval
        # Load data from directory
        _graphs = list()
        for filename in os.listdir(str(data_dir)):
            if not filename.endswith(".csv"):
                continue
            csv = pds.read_csv(str(data_dir/filename))
            graph = Graph.from_data(environment, filename=filename, df=csv)
            _graphs.append(graph)
        return _graphs

    graphs = _load_graphs()
    return graphs


def analyze_op_freq(graphs: List[Graph]):
    op_freq = defaultdict(int)
    for g in graphs:
        for node in g.nodes:
            op_freq[node.op.operator_type] += 1
    return op_freq
