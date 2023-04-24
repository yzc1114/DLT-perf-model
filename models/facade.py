from enum import Enum

from MLP import MLPModel
from base import MModule
from data.dataset import MDataset
from data.graph import Label


class ModelType(Enum):
    GBDT = 0
    GNN = 1
    MLP = 2
    PerfNet = 3
    RNN = 4
    Transformer = 5


class ModelFactory:
    @staticmethod
    def create_model(model_type: ModelType, train_ds: MDataset) -> MModule:
        sample_x = train_ds.data[0]
        sample_y = train_ds.targets[0]
        assert isinstance(sample_y, Label)
        if model_type == ModelType.MLP:
            return MLPModel(input_dimension=len(sample_x), output_dimension=len(sample_y.node_durations[0]))
        else:
            raise ValueError("Invalid model type.")
