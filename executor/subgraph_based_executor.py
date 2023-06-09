import logging
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.nn import MSELoss, LSTM

from config import Config
from data import GraphNode, Graph
from data.dataset import MDataset
from executor.base_module import MModule
from executor.executor import Executor
from executor.metric import MetricUtil
from executor.util import nested_detach, pad_np_vectors
from objects import ModelType
from .gcn import GCNLayer
from .transformer import TransformerModel


class SubgraphBasedExecutor(Executor):
    def __init__(self, conf: Config | None = None):
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

            adj_matrix = [
                [0.] * len(nodes) for _ in range(len(nodes))
            ]
            node_id_to_node = {node.node_id: node for node in nodes}
            for curr_idx, node in enumerate(nodes):
                for neighbor in node.neighbors:
                    if neighbor.node_id not in node_id_to_node:
                        continue
                    adj_idx = nodes.index(neighbor)
                    adj_matrix[curr_idx][adj_idx] = 1.

            adj_matrix = np.array(adj_matrix)
            # x
            feature = {
                "x_graph_id": graph.ID,
                "x_subgraph_feature": feature_matrix,
                "x_adj_matrix": adj_matrix
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

            x_adj_matrix = feature["x_adj_matrix"]
            x_adj_matrix = np.array(x_adj_matrix).astype(np.float32)

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
                "x_adj_matrix": x_adj_matrix
            })

            processed_labels.append({
                "y_graph_id": label["y_graph_id"],
                "y_nodes_durations": transformed_y_nodes_durations,
                "y_subgraph_durations": transformed_y_subgraph_durations
            })

        ds = MDataset(processed_features, processed_labels)
        return ds

    def _evaluate(self, model) -> Dict[str, float]:
        input_batches, output_batches, eval_loss = self._dl_evaluate_pred(model)

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
        return {"eval_loss": eval_loss, **duration_metrics}


class MLPTest_SubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.MLPTestGrouping

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {}

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {}

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_count = len(sample_x_dict["x_subgraph_feature"])
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_duration_count = len(sample_y_dict["y_nodes_durations"])
        y_nodes_duration_size = len(sample_y_dict["y_nodes_durations"][0])
        return MLPTest_SubgraphModel(x_node_feature_count,
                                     x_node_feature_size,
                                     y_nodes_duration_count,
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


class TransformerSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.Transformer

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "nhead": 8,
            "d_hid": 128,
            "nlayers": 6,
            "dropout": 0.05,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "nhead": [4, 8],
            "d_hid": [1024, 2048],
            "nlayers": [4, 6, 8],
            "dropout": [0.2, 0.5],
        }

    @staticmethod
    def grid_search_transfer_params() -> Dict[str, List]:
        return {
            "freeze_layers": [2, 3],
            "reinit_proj": [False, True]
        }

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)

        nhead = final_params["nhead"]
        while x_node_feature_size % nhead != 0:
            nhead -= 1
        if nhead != final_params["nhead"]:
            final_params["nhead"] = nhead
            logging.info(f"Transformer nhead set to {nhead}.")
            self.conf.model_params["nhead"] = nhead

        return TransformerModel(
            d_model=x_node_feature_size,
            output_d=nodes_durations_len,
            **final_params
        )


class LSTMModel(MModule):
    def __init__(self, feature_size, nodes_durations_len, num_layers, bidirectional, **kwargs):
        super().__init__(**kwargs)
        self.lstm = LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers, batch_first=True,
                         bidirectional=bidirectional)
        num_directions = 2 if bidirectional else 1
        self.project = torch.nn.Linear(in_features=feature_size * num_directions, out_features=nodes_durations_len)
        self.loss_fn = MSELoss()

    def forward(self, X):
        X = X["x_subgraph_feature"]
        out, _ = self.lstm(X)
        Y = self.project(out)
        return Y

    def compute_loss(self, outputs, Y):
        node_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, node_durations)
        return loss


class LSTMSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.LSTM

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "num_layers": 4,
            "bidirectional": True,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "num_layers": [2, 4],
            "bidirectional": [True, False],
        }

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)
        return LSTMModel(
            feature_size=x_node_feature_size,
            nodes_durations_len=y_nodes_durations_len,
            **final_params
        )


class GCNSubgraphModel(MModule):
    def __init__(self, dim_feats, dim_h, dim_out, n_layers, dropout):
        super(GCNSubgraphModel, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(dim_feats, dim_h, F.relu, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(dim_h, dim_h, F.relu, dropout))
        # output layer
        self.layers.append(GCNLayer(dim_h, dim_out, None, dropout))
        self.loss_fn = MSELoss()

    def forward(self, X):
        adj, features = X["x_adj_matrix"], X["x_subgraph_feature"]
        h = features
        for layer in self.layers:
            h = layer(adj, h)
        return h

    def compute_loss(self, outputs, Y) -> torch.Tensor:
        y_nodes_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, y_nodes_durations)
        return loss


class GCNSubgraphBasedExecutor(SubgraphBasedExecutor):
    def _init_model_type(self) -> ModelType:
        return ModelType.GCNSubgraph

    @staticmethod
    def default_model_params() -> Dict[str, Any]:
        return {
            "dim_h": None,
            "n_layers": 2,
            "dropout": 0.1,
        }

    @staticmethod
    def grid_search_model_params() -> Dict[str, List]:
        return {
            "dim_h": [32, 64],
            "n_layers": [2, 4],
            "dropout": [0.2, 0.5],
        }

    def _init_model(self) -> MModule | Any:
        sample_x_dict = self.preprocessed_train_ds.features[0]
        sample_y_dict = self.preprocessed_train_ds.labels[0]
        x_node_feature_size = len(sample_x_dict["x_subgraph_feature"][0])
        y_nodes_durations_len = len(sample_y_dict["y_nodes_durations"][0])
        model_params = self.conf.model_params
        final_params = self.default_model_params()
        for k, v in final_params.items():
            final_params[k] = model_params.get(k, v)
        if final_params["dim_h"] is None:
            final_params["dim_h"] = x_node_feature_size
        return GCNSubgraphModel(
            dim_feats=x_node_feature_size,
            dim_out=y_nodes_durations_len,
            **final_params
        )
