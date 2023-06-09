import json
from typing import Dict

import torch.cuda
from sklearn import preprocessing

from objects import GPUType, Environment, OptimizerType, ModelType


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


class TransferConfigMixin:
    def __init__(self, transfer_config_js, **kwargs):
        super().__init__(**kwargs)
        self.transfer_params: Dict | None = transfer_config_js.get("transfer_params", None)


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


class Config(DatasetConfigMixin, ModelConfigMixin, DeviceConfigMixin, TransferConfigMixin, JsonifyAble):

    @staticmethod
    def from_dict(d):
        return Config(d)

    @staticmethod
    def from_file(config_filepath):
        with open(config_filepath) as f:
            config_js = json.load(f)
        return Config.from_dict(config_js)

    def __init__(self, train_config_js):
        super().__init__(dataset_config_js=train_config_js,
                         model_config_js=train_config_js,
                         device_config_js=train_config_js,
                         transfer_config_js=train_config_js)
        # training
        self.all_seed = train_config_js.get("all_seed", 42)
        self.num_train_epochs = train_config_js.get("epochs", 50)
        self.batch_size = train_config_js.get("batch_size", 64)
        self.logging_steps = train_config_js.get("logging_steps", 100)
        self.evaluation_strategy = train_config_js.get("evaluation_strategy", "epoch")
        self.eval_steps = train_config_js.get("eval_steps", 100)
        self.load_best_model_at_end = train_config_js.get("load_best_model_at_end", True)
        # self.save_strategy = train_config_js.get("save_strategy", "epoch")
        self.optimizer_cls_str = train_config_js.get("optimizer", "Adam")
        self.optimizer_cls = OptimizerType[train_config_js.get("optimizer", "Adam")].value
        self.learning_rate = train_config_js.get("learning_rate", 1e-3)


train_configs = {
    ModelType.MLP: {
        "model": "MLP",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.PerfNet: {
        "model": "PerfNet",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.GBDT: {
        "model": "PerfNet",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": True
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.LSTM: {
        "model": "LSTM",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.GCNGrouping: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.GCNSubgraph: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam"
    },
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "optimizer": "Adam"
    },
}

transfer_configs = {
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti_pytorch_cuda118",
        "dataset_normalization": "Standard",
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "transfer_params": {
            "freeze_layers": 3,
            "reinit_proj": True
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 10,
        "optimizer": "Adam",
        "resume_from_ckpt": "2023-06-09_14-57-45/ckpt_300.pth"
    }
}
