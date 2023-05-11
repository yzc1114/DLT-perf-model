from collections import defaultdict
from typing import List, Dict, Callable

import torch
from torch.nn import MSELoss, ReLU
from transformers import TrainingArguments

from config import ModelConfigMixin
from data import FeatureKeys, Graph, MDataset
from .base import MModule, MetricUtil, MTrainer, FullGraphMetricMixin, LossUtilMixin


def MLP_init(model_config: ModelConfigMixin, train_ds: MDataset):
    sample_x_dict = train_ds.features[0]
    sample_y_dict = train_ds.labels[0]
    return MLPModel(input_dimension=len(sample_x_dict[FeatureKeys.X_OP_FEAT]),
                    output_dimension=len(sample_y_dict[FeatureKeys.Y_OP_FEAT][0]))


class MLPModel(MModule, FullGraphMetricMixin, LossUtilMixin):

    @staticmethod
    def dimension_len(t):
        return t[-1] - t[0]

    def __init__(self, input_dimension, output_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return LossUtilMixin.op_based_loss(self, self.loss_fn, inputs)

    def _full_graph_metrics(self, inputs_batches: List[Dict], outputs_batches: List, graphs: List[Graph]) -> Dict:
        return super()._op_based_full_graph_metrics(inputs_batches, outputs_batches, graphs)


class MLPTrainer(MTrainer):
    def __init__(self,
                 model_init: Callable,
                 model_params: Dict,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls,
                 resume_from_ckpt):
        self.model_params = model_params
        super().__init__(model_init,
                         args,
                         train_dataset,
                         eval_dataset,
                         optimizer_cls,
                         resume_from_ckpt)
