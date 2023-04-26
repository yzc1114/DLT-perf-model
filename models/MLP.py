from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn import MSELoss, ReLU

import numpy as np
from data import FeatureKeys, Graph
from .base import MModule


class MLPModel(MModule):
    DurationDim = 0, 3

    @staticmethod
    def dimension_len(t):
        return t[-1] - t[0]

    def __init__(self, input_dimension, output_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert output_dimension == self.dimension_len(self.DurationDim)
        self.input = torch.nn.Linear(input_dimension, 512)
        self.relu1 = ReLU()
        self.dense1 = torch.nn.Linear(512, 128)
        self.relu2 = ReLU()
        self.dense2 = torch.nn.Linear(128, 32)
        self.relu3 = ReLU()
        self.output = torch.nn.Linear(32, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = self.input(X)
        X = self.relu1(X)
        X = self.dense1(X)
        X = self.relu2(X)
        X = self.dense2(X)
        X = self.relu3(X)
        Y = self.output(X)
        return Y

    def loss(self, inputs):
        labels = inputs["labels"]
        # here, subgraph equals to op since a subgraph only contains one op
        y_op_features = labels[FeatureKeys.Y_SUBGRAPH_FEAT]
        x_op_features = inputs[FeatureKeys.X_OP_FEAT]
        outputs = self(x_op_features)
        loss = self.loss_fn(outputs, y_op_features)
        return loss, outputs

    def _full_graph_metrics(self, inputs_batches: List[Dict], outputs_batches: List, graphs: List[Graph]) -> Dict:
        assert len(inputs_batches) == len(outputs_batches)
        batches_len = len(inputs_batches)

        def compute_op_durations(outputs_):
            durations = outputs_[:, self.DurationDim[0]:self.DurationDim[1]].sum(dim=1)
            return durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = inputs_batches[idx]
            outputs = outputs_batches[idx]
            graph_ids = inputs[FeatureKeys.X_GRAPH_ID]
            op_durations = compute_op_durations(outputs)
            for i, graph_id in enumerate(graph_ids):
                op_duration = op_durations[i].item()
                graph_id_to_duration_pred[graph_id] += op_duration
        y_hat, y = list(), list()
        for graph in graphs:
            pred = graph_id_to_duration_pred[graph.ID]
            ground_truth = graph.graph_duration
            y_hat.append(pred)
            y.append(ground_truth)
        y_hat = np.array(y_hat)
        y = np.array(y)
        MRE = np.sum(np.abs(y - y_hat) / y) / len(y)
        RMSE = np.sqrt(np.sum(np.power(y - y_hat, 2)) / len(y))
        return {
            "MRE": MRE,
            "RMSE": RMSE
        }
