import logging
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.nn import MSELoss

from config import TrainConfig, EvalConfig
from data import GraphNode, Graph
from data.dataset import MDataset
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach, pad_np_vectors
from objects import ModelType
from .transformer import TransformerRegressionModel


class SubgraphBasedExecutor(Executor):
    def __init__(self, conf: TrainConfig | EvalConfig | None = None):
        super().__init__(conf)
        self.scalers: Tuple | None = None

    @staticmethod
    def subgraph_features(graph: Graph, subgraph_node_size: int = 10, **kwargs) -> Tuple[List[Dict], List[Dict]]:
        subgraphs, _ = graph.subgraphs(subgraph_node_size=subgraph_node_size)
        X, Y = list(), list()

        def subgraph_feature(nodes: List[GraphNode]):
            feature_matrix = list()
            for node in nodes:
                feature = node.op.to_feature_array(mode="complex")
                feature = np.concatenate([feature, graph.graph_meta_feature()])
                feature = np.array(feature)
                feature_matrix.append(feature)

            feature_matrix = pad_np_vectors(feature_matrix)
            feature_matrix = np.array(feature_matrix)
            # x
            feature = {
                "x_graph_id": graph.ID,
                "x_subgraph_feature": feature_matrix
            }

            # y
            attrs = ["forward_times", "backward_times", "optimizer_times"]
            subgraph_durations = tuple(
                abs(getattr(subgraph[0], attr)[0] - getattr(subgraph[-1], attr)[1])
                for attr in attrs
            )
            nodes_durations = list()
            for node in subgraph:
                node_duration_label = tuple(
                    abs(getattr(node, attr)[0] - getattr(node, attr)[1])
                    for attr in attrs
                )
                nodes_durations.append(node_duration_label)

            label = {
                "y_graph_id": graph.ID,
                "y_nodes_durations": nodes_durations,
                "y_subgraph_durations": subgraph_durations
            }

            return feature, label

        for i, subgraph in enumerate(subgraphs):
            x, y = subgraph_feature(subgraph)
            X.append(x)
            Y.append(y)

        return X, Y

    def _init_dataset(self, mode="train") -> MDataset:
        conf = self.conf
        if mode == "train":
            graphs = self.train_graphs
        else:
            graphs = self.eval_graphs

        X = list()
        Y = list()

        subgraph_feature_maxsize = 0

        for graph in graphs:
            X_, Y_ = self.subgraph_features(graph, **conf.dataset_params)
            for x in X_:
                subgraph_feature_size = len(x["x_subgraph_feature"][0])
                subgraph_feature_maxsize = max(subgraph_feature_maxsize, subgraph_feature_size)

            X.extend(X_)
            Y.extend(Y_)

        for x in X:
            x["x_subgraph_feature"] = pad_np_vectors(x["x_subgraph_feature"], maxsize=subgraph_feature_maxsize)

        dataset = MDataset(X, Y)
        return dataset

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @lru_cache(maxsize=None)
    def _get_scalers(self):
        train_ds = self.train_ds
        scaler_cls = self.conf.dataset_normalizer_cls

        x_subgraph_feature_array, y_nodes_durations_array, y_subgraph_durations_array = self._preprocess_required_data(
            ds=train_ds)

        x_subgraph_feature_scaler = scaler_cls()
        x_subgraph_feature_scaler.fit(x_subgraph_feature_array)

        y_nodes_durations_scaler = scaler_cls()
        y_nodes_durations_scaler.fit(y_nodes_durations_array)

        y_subgraph_durations_scaler = scaler_cls()
        y_subgraph_durations_scaler.fit(y_subgraph_durations_array)

        return x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler

    @staticmethod
    def _preprocess_required_data(ds: MDataset):
        x_subgraph_feature_array = list()
        y_nodes_durations_array = list()
        y_subgraph_durations_array = list()

        for data in ds:
            feature, label = data
            x_subgraph_feature = feature["x_subgraph_feature"]
            assert isinstance(x_subgraph_feature, list)
            x_subgraph_feature_array.extend(x_subgraph_feature)

            y_nodes_durations = label["y_nodes_durations"]
            assert isinstance(y_nodes_durations, list)
            y_nodes_durations_array.extend(y_nodes_durations)

            y_subgraph_durations = label["y_subgraph_durations"]
            y_subgraph_durations_array.append(y_subgraph_durations)

        x_subgraph_feature_array = np.array(x_subgraph_feature_array)
        y_nodes_durations_array = np.array(y_nodes_durations_array)
        y_subgraph_durations_array = np.array(y_subgraph_durations_array)
        return x_subgraph_feature_array, y_nodes_durations_array, y_subgraph_durations_array

    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = self._get_scalers()

        processed_features = list()
        processed_labels = list()

        for data in ds:
            feature, label = data
            x_subgraph_feature = feature["x_subgraph_feature"]
            assert isinstance(x_subgraph_feature, list)
            x_subgraph_feature = np.array(x_subgraph_feature).astype(np.float32)
            transformed_x_subgraph_feature = x_subgraph_feature_scaler.transform(x_subgraph_feature)

            y_nodes_durations = label["y_nodes_durations"]
            assert isinstance(y_nodes_durations, list)
            y_nodes_durations = np.array(y_nodes_durations).astype(np.float32)
            transformed_y_nodes_durations = y_nodes_durations_scaler.transform(y_nodes_durations)

            y_subgraph_durations = label["y_subgraph_durations"]
            y_subgraph_durations_array = (y_subgraph_durations,)
            y_subgraph_durations_array = y_subgraph_durations_scaler.transform(y_subgraph_durations_array)
            transformed_y_subgraph_durations = y_subgraph_durations_array[0]

            processed_features.append({
                "x_graph_id": feature["x_graph_id"],
                "x_subgraph_feature": transformed_x_subgraph_feature,
            })

            processed_labels.append({
                "y_graph_id": label["y_graph_id"],
                "y_nodes_durations": transformed_y_nodes_durations,
                "y_subgraph_durations": transformed_y_subgraph_durations
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def _evaluate(self, model) -> Dict[str, float]:
        input_batches, output_batches = self._dl_evaluate_pred(model)

        batches_len = len(input_batches)

        def compute_nodes_durations(outputs_):
            x_subgraph_feature_scaler, y_nodes_durations_scaler, y_subgraph_durations_scaler = self._get_scalers()
            nodes_durations = list()
            for output_ in outputs_:
                transformed: np.ndarray = y_nodes_durations_scaler.inverse_transform(output_)
                duration = transformed.sum()
                nodes_durations.append(duration)
            return nodes_durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = input_batches[idx]
            outputs = output_batches[idx]
            outputs = nested_detach(outputs)
            outputs = outputs.numpy()
            graph_ids = inputs["x_graph_id"]
            nodes_durations = compute_nodes_durations(outputs)
            for i, graph_id in enumerate(graph_ids):
                node_duration = nodes_durations[i].item()
                graph_id_to_duration_pred[graph_id] += node_duration
        duration_metrics = MetricUtil.compute_duration_metrics(self.eval_graphs, graph_id_to_duration_pred)
        return duration_metrics


class MLPTest_SubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLPTestGrouping

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_count = len(sample_x_dict["x_subgraph_feature"])
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_duration_count = len(sample_y_dict["y_nodes_durations"])
        y_nodes_duration_size = len(sample_y_dict["y_nodes_durations"][0])
        return MLPTest_SubgraphModel(x_node_feature_count, x_node_feature_size, y_nodes_duration_count,
                                     y_nodes_duration_size)


class MLPTest_SubgraphModel(MModule):

    def __init__(self, x_node_feature_count, x_node_feature_size, y_nodes_duration_count, y_nodes_duration_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.x_node_feature_count, self.x_node_feature_size, self.y_nodes_duration_count, self.y_nodes_duration_size \
            = x_node_feature_count, x_node_feature_size, y_nodes_duration_count, y_nodes_duration_size
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=self.x_node_feature_count * self.x_node_feature_size,
                                       out_features=256)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=256,
                                       out_features=512)
        self.relu2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(512, self.y_nodes_duration_count * self.y_nodes_duration_size)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_subgraph_feature"]
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu1(X)
        X = self.linear2(X)
        X = self.relu2(X)
        Y = self.output(X)
        Y = torch.reshape(Y, (-1, self.y_nodes_duration_count, self.y_nodes_duration_size))
        return Y

    def compute_loss(self, outputs, Y):
        nodes_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, nodes_durations)
        return loss


class TransformerRegressionSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLPTestGrouping

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        subgraph_durations_len = len(sample_y_dict["y_subgraph_durations"])
        model_params = self.conf.model_params
        nhead = model_params.get("nhead", 8)
        while x_node_feature_size % nhead != 0:
            nhead -= 1
        if nhead != model_params.get("nhead", 8):
            logging.info(f"Transformer nhead set to {nhead}.")
        return TransformerRegressionModel(
            d_model=x_node_feature_size,
            nhead=nhead,
            d_hid=model_params.get("d_hid", 2048),
            nlayers=model_params.get("nlayers", 6),
            dropout=model_params.get("dropout", 0.5),
            output_d=subgraph_durations_len
        )
