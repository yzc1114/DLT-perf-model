from collections import defaultdict
from typing import List, Dict, Callable

import torch
from torch.nn import MSELoss, ReLU
from transformers import TrainingArguments
from torch.optim.lr_scheduler import LambdaLR

from config import ModelConfigMixin
from data import FeatureKeys, Graph, MDataset
from .base import MModule, MetricUtil, MTrainer, FullGraphMetricMixin, LossUtilMixin


def PerfNet_init(model_config: ModelConfigMixin, train_ds: MDataset):
    # sample_x_dict = train_ds.features[0]
    sample_y_dict = train_ds.labels[0]
    return PerfNetModel(output_dimension=len(sample_y_dict[FeatureKeys.Y_OP_FEAT][0]))


class PerfNetModel(MModule, FullGraphMetricMixin):
    @staticmethod
    def dimension_len(t):
        return t[-1] - t[0]

    def __init__(self, output_dimension, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=128, kernel_size=2, bias=True, padding_mode='zeros')
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(3200, 32)
        self.relu1 = ReLU()
        self.dense2 = torch.nn.Linear(32, 64)
        self.relu2 = ReLU()
        self.dense3 = torch.nn.Linear(64, 128)
        self.relu3 = ReLU()
        self.dense4 = torch.nn.Linear(128, 256)
        self.relu4 = ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(256, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = torch.unsqueeze(X, dim=1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.relu1(X)
        X = self.dense2(X)
        X = self.relu2(X)
        X = self.dense3(X)
        X = self.relu3(X)
        X = self.dense4(X)
        X = self.relu4(X)
        X = self.dropout(X)
        Y = self.output(X)
        return Y

    def loss(self, inputs):
        return LossUtilMixin.op_based_loss(self, self.loss_fn, inputs)

    def _full_graph_metrics(self, inputs_batches: List[Dict], outputs_batches: List, graphs: List[Graph]) -> Dict:
        return super()._op_based_full_graph_metrics(inputs_batches, outputs_batches, graphs)


class PerfNetTrainer(MTrainer):
    def __init__(self,
                 model_init: Callable,
                 model_params: Dict,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls,
                 resume_from_ckpt):
        self.model_params: Dict = model_params
        super().__init__(model_init,
                         args,
                         train_dataset,
                         eval_dataset,
                         optimizer_cls,
                         resume_from_ckpt,
                         _create_optimizer_and_scheduler=self._create_optimizer_and_scheduler)


    def _create_optimizer_and_scheduler(self, hg_trainer, num_training_steps):
        init_lr = hg_trainer.args.learning_rate
        scheduler_epochs = self.model_params.get("scheduler_epochs", 30)
        optimizer = hg_trainer.optimizer_cls(hg_trainer.model.parameters(), lr=init_lr)
        def halve(epoch):
            times = epoch // scheduler_epochs
            return 0.5 ** times

        lr_scheduler = LambdaLR(optimizer, lr_lambda=halve, verbose=True)
        return optimizer, lr_scheduler
