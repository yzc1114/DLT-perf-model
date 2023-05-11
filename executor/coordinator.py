import logging
import pathlib
from itertools import product
from typing import Tuple

from transformers import TrainingArguments

from config import TrainConfig, EvalConfig, ModelConfigMixin
from data.dataset import DatasetFactory, MDataset, DatasetType
from objects import Environment
from trainers import ModelFactory, MModule, MTrainer, TrainerFactory

ckpts_dir = pathlib.Path(__file__).parent.parent / 'ckpts'
logs_dir = pathlib.Path(__file__).parent.parent / 'logs'


class Coordinator:
    @staticmethod
    def _init_model(model_config: ModelConfigMixin, train_ds: MDataset) -> MModule:
        model = ModelFactory.create_model(model_config, train_ds)
        return model

    @staticmethod
    def _init_dataset(dataset_environment: Environment,
                      dataset_type: DatasetType,
                      dataset_normalization: str,
                      dataset_dummy: bool,
                      **dataset_params) -> Tuple[MDataset, MDataset]:
        train_ds, eval_ds = \
            DatasetFactory.create_dataset(dataset_environment,
                                          dataset_normalization,
                                          dataset_type,
                                          dummy=dataset_dummy,
                                          **dataset_params)
        return train_ds, eval_ds

    @staticmethod
    def train(train_config: TrainConfig):
        product_attrs = ("dataset_normalization", "dataset_params")
        product_attr_values = list()
        for attr in product_attrs:
            attr_value = train_config.__getattribute__(attr)
            if not isinstance(attr_value, list):
                product_attr_values.append((attr_value,))
            else:
                product_attr_values.append(tuple(attr_value))
        for p in product(*product_attr_values):
            dataset_normalization = p[product_attrs.index("dataset_normalization")]
            dataset_params = p[product_attrs.index("dataset_params")]

            trainer = Coordinator.create_trainer(train_config, dataset_normalization, dataset_params)

            trainer.train()

    @staticmethod
    def create_trainer(conf, dataset_normalization=None, dataset_params=None) -> MTrainer:
        if dataset_normalization is None:
            dataset_normalization = conf.dataset_normalization
        if dataset_params is None:
            dataset_params = conf.dataset_params
        train_ds, eval_ds = Coordinator._init_dataset(
            dataset_environment=conf.dataset_environment,
            dataset_type=conf.dataset_type,
            dataset_normalization=dataset_normalization,
            dataset_dummy=conf.dataset_dummy,
            **dataset_params)

        def model_init():
            model = Coordinator._init_model(
                conf,
                train_ds
            )
            return model

        unique_path = pathlib.Path(
            conf.dataset_environment_str) / conf.model_type_str / conf.identifier()
        train_ckpts_dir = str(ckpts_dir / unique_path)
        train_logs_dir = str(logs_dir / unique_path)
        training_args = TrainingArguments(
            output_dir=train_ckpts_dir,
            num_train_epochs=getattr(conf, "num_train_epochs", 10),
            per_device_train_batch_size=conf.batch_size,
            per_device_eval_batch_size=conf.batch_size,
            logging_dir=train_logs_dir,
            logging_steps=getattr(conf, "logging_steps", 10),
            remove_unused_columns=False,
            include_inputs_for_metrics=True,
            evaluation_strategy=getattr(conf, "evaluation_strategy", "epoch"),
            load_best_model_at_end=getattr(conf, "load_best_model_at_end", True),
            save_strategy=getattr(conf, "save_strategy", "epoch"),
            learning_rate=getattr(conf, "learning_rate", 1e-3)
        )

        model_params = getattr(conf, "model_params", dict())

        trainer = TrainerFactory.create_trainer(
            model_type=conf.model_type,
            model_params=model_params,
            model_init=model_init,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            optimizer_cls=getattr(conf, "optimizer_cls", None),
            resume_from_ckpt=getattr(conf, "resume_from_ckpt", 10),
        )

        return trainer

    @staticmethod
    def eval(eval_config: EvalConfig):
        trainer = Coordinator.create_trainer(eval_config)
        metrics = trainer.evaluate()
        logging.info(metrics)
