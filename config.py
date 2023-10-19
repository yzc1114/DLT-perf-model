import json
from typing import Dict, List

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
        meta_configs = dataset_config_js.get("meta_configs", dict())
        self.dataset_environment_str: str = dataset_config_js.get("dataset_environment_str",
                                                                  "RTX2080Ti")
        self.meta_dataset_train_environment_strs: [str] = meta_configs.get("meta_dataset_train_environment_strs",
                                                                           ["RTX2080Ti"])
        self.meta_dataset_eval_environment_strs: [str] = meta_configs.get("meta_dataset_eval_environment_strs",
                                                                          ["RTX2080Ti"])
        self.dataset_gpu_type_str = self.dataset_environment_str
        self.dataset_gpu_type: GPUType = GPUType[self.dataset_gpu_type_str]
        self.meta_dataset_eval_gpu_types: List[GPUType] = [GPUType[s] for s in
                                                           self.meta_dataset_eval_environment_strs]
        self.meta_dataset_train_gpu_types: List[GPUType] = [GPUType[s] for s in
                                                            self.meta_dataset_train_environment_strs]
        self.dataset_environment: Environment = Environment(gpu_type=self.dataset_gpu_type)
        self.meta_dataset_train_environments: List[Environment] = [Environment(gpu_type=t) for t in
                                                                   self.meta_dataset_train_gpu_types]
        self.meta_dataset_eval_environments: List[Environment] = [Environment(gpu_type=t) for t in
                                                                  self.meta_dataset_eval_gpu_types]
        self.dataset_normalization = dataset_config_js.get("dataset_normalization", "Standard")
        if self.dataset_normalization == "Standard":
            self.dataset_normalizer_cls = preprocessing.StandardScaler
        elif self.dataset_normalization == "MinMax":
            self.dataset_normalizer_cls = preprocessing.MinMaxScaler
        else:
            raise ValueError(f"Invalid dataset_normalization: {self.dataset_normalization}")
        self.dataset_subgraph_node_size = dataset_config_js.get("dataset_subgraph_node_size", 10)
        self.dataset_subgraph_grouping_count = dataset_config_js.get("dataset_subgraph_grouping_count", 10)
        self.dataset_op_encoding = dataset_config_js.get("dataset_op_encoding", "frequency")
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
        meta_configs = train_config_js.get("meta_configs", 1e-3)
        self.meta_configs = {
            "learning_rate": meta_configs.get("learning_rate", 1e-3),
            "meta_learning_rate": meta_configs.get("meta_learning_rate", 1e-3),
        }


dataset_subgraph_node_sizes = [10, 20, 50]
dataset_subgraph_grouping_counts = [10, 20, 30]
dataset_op_encodings = ["frequency", "one-hot"]

train_configs = {
    ModelType.MLP: {
        "model": "MLP",
        "all_seed": 42,
        "dataset_environment_str": "RTX2080Ti",
        "dataset_normalization": "Standard",
        "dataset_op_encoding": dataset_op_encodings,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.PerfNet: {
        "model": "PerfNet",
        "dataset_environment_str": "RTX2080Ti",
        "meta_dataset_environment_strs": ["RTX2080Ti"],
        "dataset_normalization": "Standard",
        "dataset_op_encoding": dataset_op_encodings,
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
        "dataset_environment_str": "RTX2080Ti",
        "meta_dataset_environment_strs": ["RTX2080Ti"],
        "dataset_normalization": "Standard",
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": True
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.LSTM: {
        "model": "LSTM",
        "dataset_environment_str": "RTX2080Ti",
        "meta_dataset_environment_strs": ["RTX2080Ti"],
        "dataset_normalization": "Standard",
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "model_params": {
            "num_layers": 5,
            "bidirectional": True
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.GRU: {
        "model": "GRU",
        "dataset_environment_str": "RTX2080Ti",
        "meta_dataset_environment_strs": ["RTX2080Ti"],
        "dataset_normalization": "Standard",
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "model_params": {
            "num_layers": 5,
            "bidirectional": True
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.GCNGrouping: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti",
        "dataset_normalization": "Standard",
        "dataset_subgraph_grouping_count": dataset_subgraph_grouping_counts,
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.GCNSubgraph: {
        "model": "GCNGrouping",
        "dataset_environment_str": "RTX2080Ti",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "dataset_op_encoding": dataset_op_encodings,
        "all_seed": 42,
        "dataset_params": {
            "duration_summed": False
        },
        "model_params": {
            "nlayers": 6,
            "d_hid": 64,
            "dropout": 0.0
        },
        "dataset_dummy": True,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 100,
        "optimizer": "Adam",
        "meta_configs": {
            "meta_dataset_train_environment_strs": ["RTX2080Ti"],
            "meta_dataset_eval_environment_strs": ["RTX2080Ti"],
        },
    },
}

transfer_configs = {
    ModelType.Transformer: {
        "model": "Transformer",
        "dataset_environment_str": "RTX2080Ti",
        "dataset_normalization": "Standard",
        "dataset_subgraph_node_size": dataset_subgraph_node_sizes,
        "dataset_op_encoding": dataset_op_encodings,
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
