from enum import Enum

from .MLP import MLPModel
from .base import MModule
from data import MDataset, FeatureKeys


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
        sample_x_dict = train_ds.features[0]
        sample_y_dict = train_ds.labels[0]
        if model_type == ModelType.MLP:
            return MLPModel(input_dimension=len(sample_x_dict[FeatureKeys.X_OP_FEAT]),
                            output_dimension=len(sample_y_dict[FeatureKeys.Y_OP_FEAT][0]))
        else:
            raise ValueError("Invalid model type.")
