import pathlib
from itertools import product
from typing import Tuple

from transformers import TrainingArguments

from config import TrainConfig
from data.dataset import DatasetFactory, MDataset, DatasetType
from models import ModelFactory, MModule, ModelType
from objects import Environment
from trainer import MTrainer

ckpts_dir = pathlib.Path(__file__).parent.parent / 'ckpts'
logs_dir = pathlib.Path(__file__).parent.parent / 'logs'


class Coordinator:
    @staticmethod
    def _init_model(model_type: ModelType, train_ds: MDataset) -> MModule:
        model = ModelFactory.create_model(model_type, train_ds)
        return model

    @staticmethod
    def _init_dataset(dataset_environment: Environment,
                      dataset_type: DatasetType,
                      dataset_normalization: str,
                      dataset_dummy: bool,
                      **dataset_params) -> Tuple[MDataset, MDataset]:
        train_ds, eval_ds = DatasetFactory.create_dataset(dataset_environment,
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
                product_attr_values.append([attr_value])
            else:
                product_attr_values.append(attr_value)
        for p in product(product_attr_values):
            dataset_normalization = p[product_attrs.index("dataset_normalization")]
            dataset_params = p[product_attrs.index("dataset_params")]

            train_ds, eval_ds = Coordinator._init_dataset(
                dataset_environment=train_config.dataset_environment,
                dataset_type=train_config.dataset_type,
                dataset_normalization=dataset_normalization,
                dataset_dummy=train_config.dataset_dummy,
                **dataset_params)

            def model_init():
                model = Coordinator._init_model(
                    train_config.model_type,
                    train_ds
                )
                return model

            train_ckpts_dir = str(ckpts_dir / train_config.identifier())
            train_logs_dir = str(logs_dir / train_config.identifier())
            training_args = TrainingArguments(
                output_dir=train_ckpts_dir,
                num_train_epochs=getattr(train_config, "num_train_epochs"),
                per_device_train_batch_size=train_config.batch_size,
                per_device_eval_batch_size=train_config.batch_size,
                logging_dir=train_logs_dir,
                logging_steps=10,
                evaluation_strategy=train_config.evaluation_strategy,
                load_best_model_at_end=train_config.load_best_model_at_end,
                resume_from_checkpoint=train_config.resume_from_checkpoint,
                save_strategy=train_config.save_strategy,
                learning_rate=train_config.learning_rate
            )

            trainer = MTrainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                optimizer_cls=train_config.optimizer_cls,
            )

            trainer.train()

    @staticmethod
    def eval(eval_config: TrainConfig):
        train_ds, eval_ds = Coordinator._init_dataset(
            dataset_environment=eval_config.dataset_environment,
            dataset_type=eval_config.dataset_type,
            dataset_normalization=eval_config.dataset_normalization,
            dataset_dummy=eval_config.dataset_dummy,
            **eval_config.dataset_params)

        model = Coordinator._init_model(
            eval_config.model_type,
            train_ds
        )
        # TODO
        raise NotImplementedError()
