from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn import MSELoss, ReLU

import numpy as np
from data import FeatureKeys, Graph, MDataset
from .base import MModule, MetricUtil
from config import TrainConfig

def MLP_init(train_config: TrainConfig, train_ds: MDataset):
    sample_x_dict = train_ds.features[0]
    sample_y_dict = train_ds.labels[0]
    return MLPModel(input_dimension=len(sample_x_dict[FeatureKeys.X_OP_FEAT]),
             output_dimension=len(sample_y_dict[FeatureKeys.Y_OP_FEAT][0]))


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
        duration_metrics = MetricUtil.compute_duration_metrics(graphs, graph_id_to_duration_pred)
        return {
            **duration_metrics
        }
