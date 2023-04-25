import json
from typing import List

from models import ModelType
from objects import GPUType, Environment, DatasetType, OptimizerType


class DatasetConfigMixin:
    def __init__(self, config_js):
        self.dataset_environment_str: str = config_js.get("dataset_environment_str", "RTX2080Ti_pytorch_cuda118")
        self.dataset_gpu_type_str, self.dataset_framework, self.dataset_cuda_version = \
            self.dataset_environment_str.split("_")
        self.dataset_gpu_type: GPUType = GPUType[self.dataset_gpu_type_str]
        self.dataset_environment: Environment = Environment(gpu_type=self.dataset_gpu_type,
                                                            framework=self.dataset_framework,
                                                            cuda_version=self.dataset_cuda_version)
        self.dataset_normalization = config_js.get("dataset_normalization", "Standard")
        self.dataset_type_str = config_js.get("dataset_type", "OP")
        self.dataset_type: DatasetType = DatasetType[self.dataset_type_str]
        self.dataset_params = config_js.get("dataset_params", dict())
        self.dataset_train_proportion = config_js.get("train_ds_proportion", 0.7)
        self.dataset_seed = config_js.get("dataset_seed", 42)
        self.dataset_dummy = config_js.get("dataset_dummy", False)


class TrainConfig(DatasetConfigMixin):
    def __init__(self, train_config_js):
        super().__init__(train_config_js)
        # training
        self.model_type_str = train_config_js.get("model", "MLP")
        self.model_type: ModelType = ModelType[self.model_type_str]
        self.train_seed = train_config_js.get("train_seed", 42)
        self.dataset_sample_seed = train_config_js.get("dataset_sample_seed", 42)
        self.num_train_epochs = train_config_js.get("num_train_epochs", 100)
        self.batch_size = train_config_js.get("batch_size", 64)
        self.logging_steps = train_config_js.get("logging_steps", 10)
        self.evaluation_strategy = train_config_js.get("evaluation_strategy", "epoch")
        self.eval_steps = train_config_js.get("eval_steps", 50)
        self.load_best_model_at_end = train_config_js.get("load_best_model_at_end", True)
        self.resume_from_checkpoint = train_config_js.get("resume_from_checkpoint", None)
        self.save_strategy = train_config_js.get("save_strategy", "epoch")
        self.optimizer_cls_str = train_config_js.get("optimizer", "Adam")
        self.optimizer_cls = OptimizerType[train_config_js.get("optimizer", "Adam")].value
        self.learning_rate = train_config_js.get("learning_rate", 1e-3)

    def identifier(self) -> str:
        dataset_param_list = list()
        for k, v in self.dataset_params.items():
            dataset_param_list.append(f"{k}_{v}")
        dataset_param_str = "|".join(dataset_param_list)
        return f"{self.dataset_environment_str}|{self.dataset_normalization}|{self.batch_size}|{self.learning_rate}|{dataset_param_str}"


class EvalConfig(DatasetConfigMixin):
    def __init__(self, eval_config_js):
        super().__init__(eval_config_js)
        # evaluating
        self.model_type_str = eval_config_js.get("model_type_str", "MLP")
        self.model_type: ModelType = ModelType[self.model_type_str]

    def identifier(self) -> str:
        dataset_param_list = list()
        for k, v in self.dataset_params.items():
            dataset_param_list.append(f"{k}_{v}")
        dataset_param_str = "|".join(dataset_param_list)
        return f"{self.dataset_environment_str}|{self.dataset_normalization}|{dataset_param_str}"


class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_js = json.load(f)
        self.config_type = self.config_js["config_type"]

        def load_configs(config_type):
            config_cls = {
                "train": TrainConfig,
                "eval": EvalConfig
            }
            assert config_type in self.config_js
            return [config_cls[config_type](sub_config_js) for sub_config_js in self.config_js[config_type]]

        if self.config_type == "train":
            self.train_configs: List[TrainConfig] = load_configs(self.config_type)
        elif self.config_type == "eval":
            self.eval_configs: List[EvalConfig] = load_configs(self.config_type)
        else:
            raise ValueError(f"Invalid config type: {self.config_type}.")
