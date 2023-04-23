import torch

from typing import List
from graph import Graph, Label
from .base import PerfPredictor

dummy_graph = Graph.from_data(dummy=True)
sample_X, sample_labels = dummy_graph.Serial_feature_extractor.node_features()
sample_x, sample_label = sample_X[0], sample_labels[0]

def MLP_label(label: Label) -> List:
    return list(label.node_durations[0])


class MLPPredictor(PerfPredictor):
    def predict(self, graph: Graph) -> float:
        pass


# class MLPModel(torch.nn.Module):
#     def __init__(self):
#         self.dense1 = torch.nn.Linear(512, 128)
#         self.dense2 = torch.nn.Linear(128, 32)
#         self.dense_out = torch.nn.Linear(32, 128)
#
#         # 512, 128, 16, and 1
#     def forward(self, X):
