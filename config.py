import json
from typing import List

import torch.cuda
from sklearn import preprocessing

from objects import GPUType, Environment, DatasetType, OptimizerType, ModelType


class JsonifyAble:
    def to_dict(self):
        attr_names = dir(self)
        d = dict()
        for attr_name in attr_names:
            if attr_name.startswith("__"):
                continue
            attr = self.__getattribute__(attr_name)
            if isinstance(attr, int) or \
                    isinstance(attr, str) or \
                    isinstance(attr, dict) or \
                    isinstance(attr, list) or \
                    isinstance(attr, bool) or \
                    attr is None:
                d[attr_name] = attr
        return d


class DatasetConfigMixin:
    def __init__(self, dataset_config_js, **kwargs):
        super().__init__(**kwargs)
        self.dataset_environment_str: str = dataset_config_js.get("dataset_environment_str",
                                                                  "RTX2080Ti_pytorch_cuda118")
        self.dataset_gpu_type_str, self.dataset_framework, self.dataset_cuda_version = \
            self.dataset_environment_str.split("_")
        self.dataset_gpu_type: GPUType = GPUType[self.dataset_gpu_type_str]
        self.dataset_environment: Environment = Environment(gpu_type=self.dataset_gpu_type,
                                                            framework=self.dataset_framework,
                                                            cuda_version=self.dataset_cuda_version)
        self.dataset_normalization = dataset_config_js.get("dataset_normalization", "Standard")
        if self.dataset_normalization == "Standard":
            self.dataset_normalizer_cls = preprocessing.StandardScaler
        elif self.dataset_normalization == "MinMax":
            self.dataset_normalizer_cls = preprocessing.MinMaxScaler
        else:
            raise ValueError(f"Invalid dataset_normalization: {self.dataset_normalization}")
        self.dataset_type_str = dataset_config_js.get("dataset_type", "OP")
        self.dataset_type: DatasetType = DatasetType[self.dataset_type_str]
        self.dataset_params = dataset_config_js.get("dataset_params", dict())
        self.dataset_train_proportion = dataset_config_js.get("train_ds_proportion", 0.7)
        self.dataset_dummy = dataset_config_js.get("dataset_dummy", False)

    def identifier(self) -> str:
        dataset_param_list = list()
        for k, v in self.dataset_params.items():
            dataset_param_list.append(f"{k}_{v}")
        s = f"{self.dataset_normalization}"
        if len(dataset_param_list) > 0:
            dataset_param_str = "|".join(dataset_param_list)
            s += f"|{dataset_param_str}"
        return s


class ModelConfigMixin:
    def __init__(self, model_config_js, **kwargs):
        super().__init__(**kwargs)
        self.model_type_str = model_config_js.get("model", "MLP")
        self.model_type: ModelType = ModelType[self.model_type_str]
        self.model_params = model_config_js.get("model_params", dict())
        self.resume_from_ckpt = model_config_js.get("resume_from_ckpt", None)


class DeviceConfigMixin:
    def __init__(self, device_config_js, **kwargs):
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.device = device_config_js.get("device", "cuda:0")
        else:
            self.device = device_config_js.get("device_type", "cpu")


class TrainConfig(DatasetConfigMixin, ModelConfigMixin, DeviceConfigMixin, JsonifyAble):
    def __init__(self, train_config_js):
        super().__init__(dataset_config_js=train_config_js,
                         model_config_js=train_config_js,
                         device_config_js=train_config_js)
        # training
        self.all_seed = train_config_js.get("all_seed", 42)
        self.num_train_epochs = train_config_js.get("num_train_epochs", 100)
        self.batch_size = train_config_js.get("batch_size", 64)
        self.logging_steps = train_config_js.get("logging_steps", 100)
        self.evaluation_strategy = train_config_js.get("evaluation_strategy", "epoch")
        self.eval_steps = train_config_js.get("eval_steps", 50)
        self.load_best_model_at_end = train_config_js.get("load_best_model_at_end", True)
        # self.save_strategy = train_config_js.get("save_strategy", "epoch")
        self.optimizer_cls_str = train_config_js.get("optimizer", "Adam")
        self.optimizer_cls = OptimizerType[train_config_js.get("optimizer", "Adam")].value
        self.learning_rate = train_config_js.get("learning_rate", 1e-3)


class EvalConfig(DatasetConfigMixin, ModelConfigMixin, DeviceConfigMixin, JsonifyAble):
    def __init__(self, eval_config_js):
        super().__init__(model_config_js=eval_config_js,
                         dataset_config_js=eval_config_js,
                         device_config_js=eval_config_js)
        self.all_seed = eval_config_js.get("all_seed", 42)
        self.batch_size: int = eval_config_js.get("batch_size", 64)


class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_js = json.load(f)

        def load_configs(config_type):
            config_cls = {
                "train": TrainConfig,
                "eval": EvalConfig
            }
            return [config_cls[config_type](sub_config_js) for sub_config_js in self.config_js[config_type]]

        self.train_configs: List[TrainConfig] = list()
        self.eval_configs: List[EvalConfig] = list()
        if "train" in self.config_js:
            self.train_configs = load_configs("train")
        if "eval" in self.config_js:
            self.eval_configs = load_configs("eval")
