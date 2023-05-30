from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.nn import MSELoss

from config import TrainConfig, EvalConfig
from data.dataset import MDataset
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach, pad_np_vectors
from objects import ModelType


class GroupingBasedExecutor(Executor):
    def __init__(self, conf: TrainConfig | EvalConfig | None = None):
        super().__init__(conf)
        self.scalers: Tuple | None = None

    @staticmethod
    def full_graph_feature(graph, subgraph_count: int = 10, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        subgraphs, node_id_to_group_idx = graph.subgraphs(subgraph_count=subgraph_count)

        feature_matrix = list()
        for subgraph in subgraphs:
            subgraph_features = list()
            for node in subgraph:
                node_feature = np.array(node.op.to_feature_array(mode="complex"))
                subgraph_features.append(node_feature)
            if len(subgraph_features) == 0:
                feature_matrix.append(np.zeros(1))
                continue
            subgraph_features = pad_np_vectors(subgraph_features)
            feature = np.sum(subgraph_features, axis=0)
            feature = np.append(feature, len(subgraph))
            feature_matrix.append(feature)

        adjacency_matrix = list()
        for i, subgraph in enumerate(subgraphs):
            vector = np.zeros(len(subgraphs) + 1)
            for node in subgraph:
                neighbor_group_indices = list()
                for neighbor in node.neighbors:
                    neighbor_group_idx = node_id_to_group_idx[neighbor.node_id]
                    if neighbor_group_idx != i:
                        neighbor_group_indices.append(neighbor_group_idx)
                for idx in neighbor_group_indices:
                    vector[idx] = 1
            adjacency_matrix.append(vector)

        optimizer_node_feature = np.array(graph.graph_meta_feature())
        feature_matrix.append(optimizer_node_feature)
        adjacency_matrix.append(np.zeros(len(adjacency_matrix[0])))

        feature_matrix = pad_np_vectors(feature_matrix)

        def pad_matrix(matrix):
            if len(matrix) < subgraph_count + 1:  # optimizer_feature
                matrix.extend([np.zeros(matrix[0].shape) for _ in range(subgraph_count + 1 - len(matrix))])

        pad_matrix(feature_matrix)
        pad_matrix(adjacency_matrix)
        feature_matrix = np.array(feature_matrix)
        adjacency_matrix = np.array(adjacency_matrix)

        x = {
            "x_graph_id": graph.ID,
            "x_feature_matrix": feature_matrix,
            "x_adjacency_metrix": adjacency_matrix,
        }
        y = {
            "y_graph_id": graph.ID,
            "y_graph_duration": (graph.graph_duration,)
        }
        return x, y

    def _init_dataset(self, mode="train") -> MDataset:
        conf = self.conf
        if mode == "train":
            graphs = self.train_graphs
        else:
            graphs = self.eval_graphs

        X = list()
        Y = list()

        feature_matrix_maxsize = 0
        adjacency_matrix_maxsize = 0

        for graph in graphs:
            x, y = self.full_graph_feature(graph, **conf.dataset_params)
            feature_matrix_size = len(x["x_feature_matrix"][0])
            adjacency_matrix_size = len(x["x_adjacency_metrix"][0])
            feature_matrix_maxsize = max(feature_matrix_maxsize, feature_matrix_size)
            adjacency_matrix_maxsize = max(adjacency_matrix_maxsize, adjacency_matrix_size)

            X.append(x)
            Y.append(y)
        for x in X:
            x["x_feature_matrix"] = pad_np_vectors(x["x_feature_matrix"], maxsize=feature_matrix_maxsize)
            x["x_adjacency_metrix"] = pad_np_vectors(x["x_adjacency_metrix"], maxsize=adjacency_matrix_maxsize)

        dataset = MDataset(X, Y)
        return dataset

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @lru_cache(maxsize=None)
    def _get_scalers(self):
        train_ds = self.train_ds
        scaler_cls = self.conf.dataset_normalizer_cls
        graph_feature_array = list()
        y_array = list()

        for data in train_ds:
            feature, label = data
            x_feature_matrix = feature["x_feature_matrix"]
            assert isinstance(x_feature_matrix, list)
            graph_feature_array.extend(x_feature_matrix)
            y_array.append(label["y_graph_duration"])

        graph_feature_array = np.array(graph_feature_array)
        y_array = np.array(y_array)

        graph_feature_scaler = scaler_cls()
        graph_feature_scaler.fit(graph_feature_array)

        y_scaler = scaler_cls()
        y_scaler.fit(y_array)
        return graph_feature_scaler, y_scaler

    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        y_array = list()

        graph_feature_scaler, y_scaler = self._get_scalers()
        graph_feature_arrays = list()
        for data in ds:
            feature, label = data
            # x. transform for each x feature matrix. do not merge them.
            x_feature_matrix = feature["x_feature_matrix"]
            x_feature_matrix = np.array(x_feature_matrix).astype(np.float32)
            graph_feature_array = graph_feature_scaler.transform(x_feature_matrix)
            graph_feature_arrays.append(graph_feature_array)
            # y. transform altogether
            y_array.append(label["y_graph_duration"])

        y_array = np.array(y_array).astype(np.float32)
        y_array = y_scaler.transform(y_array)

        processed_features = list()
        processed_labels = list()
        for i, data in enumerate(ds):
            feature, label = data
            processed_features.append({
                "x_graph_id": feature["x_graph_id"],
                "x_feature_matrix": graph_feature_arrays[i],
                "x_adjacency_metrix": feature["x_adjacency_metrix"]
            })
            processed_labels.append({
                "y_graph_id": label["y_graph_id"],
                "y_graph_duration": y_array[i],
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def _evaluate(self, model) -> Dict[str, float]:
        input_batches, output_batches = self._dl_evaluate_pred(model)

        batches_len = len(input_batches)

        def compute_graph_duration(_logits):
            _, y_scaler = self._get_scalers()
            transformed: np.ndarray = y_scaler.inverse_transform(_logits)
            duration_dim = (0, 1)
            durations = transformed[:, duration_dim[0]:duration_dim[1]].sum(axis=1)
            return durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = input_batches[idx]
            logits = output_batches[idx]
            logits = nested_detach(logits)
            logits = logits.numpy()
            graph_ids = inputs["x_graph_id"]
            graph_durations = compute_graph_duration(logits)
            for i, graph_id in enumerate(graph_ids):
                graph_duration = graph_durations[i].item()
                graph_id_to_duration_pred[graph_id] = graph_duration
        duration_metrics = MetricUtil.compute_duration_metrics(self.eval_graphs, graph_id_to_duration_pred)
        return duration_metrics


class MLPTest_GroupingBasedExecutor(GroupingBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLPTestGrouping

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        shape = len(sample_x_dict["x_feature_matrix"]), len(sample_x_dict["x_feature_matrix"][0])
        return MLPTest_GroupingModel(input_shape=shape,
                                     output_dimension=len(sample_y_dict["y_graph_duration"]))


class MLPTest_GroupingModel(MModule):

    def __init__(self, input_shape, output_dimension, **kwargs):
        super().__init__(**kwargs)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=input_shape[0] * input_shape[1], out_features=128)
        self.output = torch.nn.Linear(128, output_dimension)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_feature_matrix"]
        X = self.flatten(X)
        X = self.linear1(X)
        Y = self.output(X)
        return Y

    def compute_loss(self, outputs, Y):
        graph_duration = Y["y_graph_duration"]
        loss = self.loss_fn(outputs, graph_duration)
        return loss
