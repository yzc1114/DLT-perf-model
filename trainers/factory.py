from typing import Callable
from typing import Union, Dict

from torch.optim import Optimizer
from transformers import TrainingArguments

from config import ModelConfigMixin
from data import MDataset
from objects import ModelType
from .GBDT import GBDT_init, GBDTTrainer
from .MLP import MLP_init, MLPTrainer
from .PerfNet import PerfNet_init, PerfNetTrainer


class ModelFactory:
    init_funcs = {
        ModelType.MLP: MLP_init,
        ModelType.GBDT: GBDT_init,
        ModelType.PerfNet: PerfNet_init
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
        ModelType.PerfNet: PerfNetTrainer
    }

    @staticmethod
    def create_trainer(model_type: ModelType,
                       model_params: Dict,
                       model_init: Callable,
                       args: TrainingArguments,
                       train_dataset: MDataset,
                       eval_dataset: MDataset,
                       optimizer_cls: Optimizer,
                       resume_from_ckpt: Union[bool, str] | None):
        trainer_cls = TrainerFactory.trainer_classes[model_type]
        trainer = trainer_cls(model_init, model_params, args, train_dataset, eval_dataset, optimizer_cls, resume_from_ckpt)
        return trainer
