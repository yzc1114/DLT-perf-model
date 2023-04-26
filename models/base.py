from abc import ABC
from abc import abstractmethod
from data import MDataset, DatasetFactory
from typing import List, Dict, Tuple, Any

import torch.nn


class MModule(torch.nn.Module, ABC):
    @abstractmethod
    def loss(self, inputs) -> Tuple[torch.Tensor, Any]:
        pass

    def full_graph_metrics(self, inputs_batches: List[List], outputs_batches: List, eval_dataset: MDataset) -> Dict:
        graphs = DatasetFactory.graphs_cache[eval_dataset.graphs_cache_key]
        return self._full_graph_metrics(inputs_batches, outputs_batches, graphs)

    @abstractmethod
    def _full_graph_metrics(self, inputs_batches, outputs_batches, graphs) -> Dict:
        pass
