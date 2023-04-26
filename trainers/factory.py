from enum import Enum

from .MLP import MLP_init
from .GBDT import GBDT_init
from .base import MModule
from data import MDataset, FeatureKeys
from config import TrainConfig


class ModelType(Enum):
    GBDT = 0
    GNN = 1
    MLP = 2
    PerfNet = 3
    RNN = 4
    Transformer = 5


class ModelFactory:

    init_funcs = {
        ModelType.MLP: MLP_init,
        ModelType.GBDT: GBDT_init,
    }

    @staticmethod
    def create_model(train_config: TrainConfig, train_ds: MDataset):
        if train_config.model_type in ModelFactory.init_funcs:
            func = ModelFactory.init_funcs[train_config.model_type]
            return func(train_config, train_ds)
        raise ValueError("Invalid model type.")
