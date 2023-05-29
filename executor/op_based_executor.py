import logging
import pathlib
import pickle
import time
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import count
from typing import List, Dict
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn
import torch.optim
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss, ReLU
from torch.utils.data import DataLoader

from config import TrainConfig, EvalConfig
from data.dataset import MDataset
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach
from objects import ModelType


class OPBasedExecutor(Executor):
    def __init__(self, conf: TrainConfig | EvalConfig | None = None):
        super().__init__(conf)
        self.scalers: Tuple | None = None

    @staticmethod
    def node_features(g,
                      op_type_encoding="one-hot",
                      mode="complex",
                      encode_hyper_to_node: bool = True,
                      duration_summed: bool = False) -> Tuple[
        List[Dict], List[Dict]]:
        X, Y = list(), list()
        optimizer_feature = g.optimizer_type.encode()
        for i, node in enumerate(g.nodes):
            x_op_feature = node.op.to_feature_array(op_type_encoding, mode)
            if encode_hyper_to_node:
                x_op_feature = np.append(x_op_feature, optimizer_feature)
            x = {
                "x_op_feature": x_op_feature
            }
            attrs = ["forward_times", "backward_times", "optimizer_times"]
            node_durations = tuple(
                abs(getattr(node, attr)[0] - getattr(node, attr)[1])
                for attr in attrs
            )
            node_durations = (np.sum(node_durations),) if duration_summed else node_durations

            x["x_id"] = i
            x["x_graph_id"] = g.ID
            y = {"y_node_durations": node_durations, "y_id": i, "y_graph_id": g.ID}
            X.append(x)
            Y.append(y)
        return X, Y

    def _init_dataset(self, mode="train") -> MDataset:
        conf = self.conf
        if mode == "train":
            graphs = self.train_graphs
        else:
            graphs = self.eval_graphs

        op_X, op_Y = list(), list()
        data_idx_to_graph = dict()
        counter = iter(count())
        op_feature_len = 0

        for graph in graphs:
            X, Y = self.node_features(graph, **conf.dataset_params)
            for x in X:
                op_feature_len = max(op_feature_len, len(x["x_op_feature"]))
            op_X.extend(X)
            op_Y.extend(Y)
            for i in range(len(X)):
                data_idx_to_graph[next(counter)] = graph
        for x in op_X:
            v = x["x_op_feature"]
            x["x_op_feature"] = np.pad(v, (0, op_feature_len - v.size))

        dataset = MDataset(op_X, op_Y)
        return dataset

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @lru_cache(maxsize=None)
    def _get_scalers(self):
        train_ds = self.train_ds
        scaler_cls = self.conf.dataset_normalizer_cls
        op_feature_array = list()
        y_array = list()

        for data in train_ds:
            feature, label = data
            op_feature_array.append(feature["x_op_feature"])
            y_array.append(label["y_node_durations"])

        op_feature_array = np.array(op_feature_array)
        y_array = np.array(y_array)

        op_feature_scaler = scaler_cls()
        op_feature_scaler.fit(op_feature_array)

        y_scaler = scaler_cls()
        y_scaler.fit(y_array)
        return op_feature_scaler, y_scaler

    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        op_feature_array = list()
        y_array = list()

        for data in ds:
            feature, label = data
            op_feature_array.append(feature["x_op_feature"])
            y_array.append(label["y_node_durations"])

        op_feature_array = np.array(op_feature_array).astype(np.float32)
        y_array = np.array(y_array).astype(np.float32)

        op_feature_scaler, y_scaler = self._get_scalers()
        op_feature_array = op_feature_scaler.transform(op_feature_array)
        y_array = y_scaler.transform(y_array)

        processed_features = list()
        processed_labels = list()
        for i, data in enumerate(ds):
            feature, label = data
            processed_features.append({
                "x_id": feature["x_id"],
                "x_graph_id": feature["x_graph_id"],
                "x_op_feature": op_feature_array[i],
            })
            processed_labels.append({
                "y_id": label["y_id"],
                "y_graph_id": label["y_graph_id"],
                "y_node_durations": y_array[i]
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def _evaluate(self, model) -> Dict[str, float]:
        processed_eval_ds = self.preprocessed_eval_ds
        dl = DataLoader(processed_eval_ds, batch_size=self.conf.batch_size, shuffle=False)
        input_batches = list()
        output_batches = list()
        for data in dl:
            features, _ = data
            with torch.no_grad():
                outputs = model(features)
            input_batches.append(features)
            output_batches.append(outputs)

        batches_len = len(input_batches)

        def compute_op_durations(_logits):
            _, y_scaler = self._get_scalers()
            transformed: np.ndarray = y_scaler.inverse_transform(_logits)
            duration_dim = (0, 3)
            durations = transformed[:, duration_dim[0]:duration_dim[1]].sum(axis=1)
            return durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = input_batches[idx]
            logits = output_batches[idx]
            logits = nested_detach(logits)
            logits = logits.numpy()
            graph_ids = inputs["x_graph_id"]
            op_durations = compute_op_durations(logits)
            for i, graph_id in enumerate(graph_ids):
                op_duration = op_durations[i].item()
                graph_id_to_duration_pred[graph_id] += op_duration
        duration_metrics = MetricUtil.compute_duration_metrics(self.eval_graphs, graph_id_to_duration_pred)
        return duration_metrics


class MLPModel(MModule):

    @staticmethod
    def dimension_len(t):
        return t[-1] - t[0]

    def __init__(self, input_dimension, output_dimension, **kwargs):
        super().__init__(**kwargs)
        self.input = torch.nn.Linear(input_dimension, 512)
        self.relu1 = ReLU()
        self.dense1 = torch.nn.Linear(512, 128)
        self.relu2 = ReLU()
        self.dense2 = torch.nn.Linear(128, 32)
        self.relu3 = ReLU()
        self.output = torch.nn.Linear(32, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_op_feature"]
        X = self.input(X)
        X = self.relu1(X)
        X = self.dense1(X)
        X = self.relu2(X)
        X = self.dense2(X)
        X = self.relu3(X)
        Y = self.output(X)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_node_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


class PerfNetModel(MModule):
    @staticmethod
    def dimension_len(t):
        return t[-1] - t[0]

    def __init__(self, output_dimension, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=128, kernel_size=2, bias=True, padding_mode='zeros')
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(3200, 32)
        self.relu1 = ReLU()
        self.dense2 = torch.nn.Linear(32, 64)
        self.relu2 = ReLU()
        self.dense3 = torch.nn.Linear(64, 128)
        self.relu3 = ReLU()
        self.dense4 = torch.nn.Linear(128, 256)
        self.relu4 = ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(256, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_op_feature"]
        X = torch.unsqueeze(X, dim=1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.relu1(X)
        X = self.dense2(X)
        X = self.relu2(X)
        X = self.dense3(X)
        X = self.relu3(X)
        X = self.dense4(X)
        X = self.relu4(X)
        X = self.dropout(X)
        Y = self.output(X)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_node_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


class MLP_OPBasedExecutor(OPBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLP

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_eval_ds.labels[0]
        return MLPModel(input_dimension=len(sample_x_dict["x_op_feature"]),
                        output_dimension=len(sample_y_dict["y_node_durations"]))


class PerfNet_OPBasedExecutor(OPBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.PerfNet

    def _init_model(self) -> MModule | Any:
        processed_train_ds = self.preprocessed_train_ds
        sample_y_dict = processed_train_ds.labels[0]
        return PerfNetModel(output_dimension=len(sample_y_dict["y_node_duration"]))


class GBDT_OPBasedExecutor(OPBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.GBDT

    def _check_params(self):
        assert self.conf.dataset_params["duration_summed"]

    def _init_model(self) -> MModule | Any:
        conf = self.conf
        config_params = conf.model_params
        model_params = {
            "n_estimators": config_params.get("n_estimators", 500),
            "max_depth": config_params.get("max_depth", 4),
            "min_samples_split": config_params.get("min_samples_split", 4),
            "learning_rate": config_params.get("learning_rate", 0.001),
            "loss": "squared_error",
        }
        reg = ensemble.GradientBoostingRegressor(**model_params)
        return reg

    @staticmethod
    def dataset_to_samples(dataset: MDataset):
        samples = list(dataset)
        X = defaultdict(list)
        Y = defaultdict(list)
        for x, y in samples:
            assert isinstance(x, dict) and isinstance(y, dict)

            def collate(d: Dict, C):
                for k, v in d.items():
                    C[k].append(v)

            collate(x, X)
            collate(y, Y)
        return X, Y

    @staticmethod
    def _save_filename(loss):
        return "checkpoint_%.2f.pickle" % loss

    @staticmethod
    def _loss_from_filename(filename: str) -> float:
        return float(filename[:filename.index(".")].split("_")[-1])

    def train(self):
        logging.info(f"{self.model_type} starts training.")
        ds = self.preprocessed_train_ds
        X, Y = self.dataset_to_samples(ds)
        model = self.init_model()
        start = time.time_ns()
        X_OP = X["x_op_feature"]
        Y_dur = Y["y_node_durations"]
        model.fit(X_OP, Y_dur)
        end = time.time_ns()
        second_dur = (end - start) / 1e9
        logging.info(f"{self.model_type} ends training for {second_dur} seconds.")
        metrics = self._evaluate(model)
        eval_loss = metrics["eval_loss"]
        self.train_records["eval_metrics"] = {
            "metrics": metrics,
            "duration": second_dur
        }
        self.save_model(model=model, curr_steps=0, curr_loss_value=eval_loss)

    @staticmethod
    def _save_ckpt_to(model, filepath):
        with open(pathlib.Path(filepath), "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def _load_ckpt(ckpt_filepath) -> MModule | Any:
        with open(pathlib.Path(ckpt_filepath), "rb") as f:
            model = pickle.load(f)
            return model

    def _evaluate(
            self,
            model
    ) -> Dict[str, float]:
        ds = self.preprocessed_eval_ds
        X, Y = self.dataset_to_samples(ds)
        X_OP = X["x_op_feature"]
        Y_dur = Y["y_node_durations"]
        Y_pred = model.predict(X_OP)

        Y_dur = np.array(Y_dur).reshape(-1, 1)
        Y_pred = np.array(Y_pred).reshape(-1, 1)
        _, y_scaler = self._get_scalers()
        transformed_Y: np.ndarray = y_scaler.inverse_transform(Y_dur).reshape(-1)
        transformed_Y_pred: np.ndarray = y_scaler.inverse_transform(Y_pred).reshape(-1)

        mse = mean_squared_error(transformed_Y, transformed_Y_pred)

        dataset_len = len(X)

        op_durations = transformed_Y_pred
        graph_id_to_duration_pred = defaultdict(int)
        for i in range(dataset_len):
            graph_id = X["x_graph_id"][i]
            op_pred = op_durations[i]
            graph_id_to_duration_pred[graph_id] += op_pred

        graphs = self.eval_graphs
        duration_metrics = MetricUtil.compute_duration_metrics(graphs, graph_id_to_duration_pred)
        return {
            "eval_loss": mse,
            **duration_metrics
        }
