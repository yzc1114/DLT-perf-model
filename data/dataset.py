import pathlib
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict

from torch.utils.data import Dataset

from objects import Environment
from .graph import Graph

datasets_path = str(pathlib.Path(__file__) / "datasets")


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
            return list(Graph.from_data(None, dummy=True, seed=seed) for seed in range(500))
        data_dir = pathlib.Path(datasets_path) / f"{environment}"
        # Load data from directory
        return list()

    graphs = _load_graphs()
    return graphs


def analyze_op_freq(graphs: List[Graph]):
    op_freq = defaultdict(int)
    for g in graphs:
        for node in g.nodes:
            op_freq[node.op.operator_type] += 1
    return op_freq
