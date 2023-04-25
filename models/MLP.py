import torch
from torch.nn import MSELoss

from .base import MModule
from data import FeatureKeys


class MLPModel(MModule):
    def __init__(self, input_dimension, output_dimension, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = torch.nn.Linear(input_dimension, 512)
        self.dense1 = torch.nn.Linear(512, 128)
        self.dense2 = torch.nn.Linear(128, 32)
        self.output = torch.nn.Linear(32, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = self.input(X)
        X = self.dense1(X)
        X = self.dense2(X)
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
