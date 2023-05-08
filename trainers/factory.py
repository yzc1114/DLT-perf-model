from typing import Callable

from torch.optim import Optimizer
from transformers import TrainingArguments

from config import ModelConfigMixin
from data import MDataset
from objects import ModelType
from .GBDT import GBDT_init, GBDTTrainer
from .MLP import MLP_init, MLPTrainer


class ModelFactory:
    init_funcs = {
        ModelType.MLP: MLP_init,
        ModelType.GBDT: GBDT_init,
    }

    @staticmethod
    def create_model(model_config: ModelConfigMixin, train_ds: MDataset):
        if model_config.model_type in ModelFactory.init_funcs:
            func = ModelFactory.init_funcs[model_config.model_type]
            return func(model_config, train_ds)
        raise ValueError("Invalid model type.")


class TrainerFactory:
    trainer_classes = {
        ModelType.MLP: MLPTrainer,
        ModelType.GBDT: GBDTTrainer,
    }

    @staticmethod
    def create_trainer(model_type: ModelType,
                       model_init: Callable,
                       args: TrainingArguments,
                       train_dataset: MDataset,
                       eval_dataset: MDataset,
                       optimizer_cls: Optimizer):
        trainer_cls = TrainerFactory.trainer_classes[model_type]
        trainer = trainer_cls(model_init, args, train_dataset, eval_dataset, optimizer_cls)
        return trainer
