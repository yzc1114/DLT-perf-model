import random
import string
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from objects import GPUType, OptimizerType
from .op import Operator, OperatorType
from .util import pad_np_vectors


class FeatureKeys:
    FEAT_SUFFIX = "_feature"

    X_ID = "x_id"
    X_GRAPH_ID = "x_graph_id"
    X_OP_FEAT = f"x_op_{FEAT_SUFFIX}"
    X_SUBGRAPH_ADJACENCY_FEAT = f"x_subgraph_adjacency"
    X_SUBGRAPH_FEAT = f"x_subgraph_{FEAT_SUFFIX}"

    Y_ID = "y_id"
    Y_GRAPH_ID = "y_graph_id"
    Y_SUBGRAPH_FEAT = f"y_subgraph_{FEAT_SUFFIX}"
    Y_OP_FEAT = f"y_op_{FEAT_SUFFIX}"

class GraphNode:
    def __init__(self,
                 node_id: int,
                 op: Operator,
                 forward_times: Tuple[float, float],
                 backward_times: Tuple[float, float],
                 optimizer_times: Tuple[float, float],
                 neighbors: Optional[List[Operator]] = None):
        self.node_id: int = node_id
        self.op: Operator = op
        self.forward_times: Tuple[float, float] = forward_times
        self.backward_times: Tuple[float, float] = backward_times
        self.optimizer_times: Tuple[float, float] = optimizer_times
        self.neighbors: List[GraphNode] = list(
        ) if neighbors is None else neighbors

    def add_neighbors(self, *neighbors: 'GraphNode'):
        self.neighbors.extend(neighbors)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id

    def __ne__(self, other):
        return not self.__eq__(other)


DurationTuple = namedtuple(typename="DurationTuple", field_names=["forward", "backward", "optimization"])


class Graph:
    def __init__(self,
                 ID: Optional[str],
                 gpu_type: Optional[GPUType],
                 batch_size: Optional[int],
                 optimizer_type: Optional[OptimizerType],
                 nodes: Optional[List[GraphNode]],
                 root_node: Optional[GraphNode]):
        self.ID: Optional[str] = ID
        self.gpu_type: Optional[GPUType] = gpu_type
        self.batch_size: Optional[int] = batch_size
        self.optimizer_type: Optional[OptimizerType] = optimizer_type
        self.nodes: List[GraphNode] = list() if nodes is None else nodes
        self.root_node: Optional[GraphNode] = root_node

        self.GNN_based_feature_extractor: Graph.GNNBasedFeatureExtractor = Graph.GNNBasedFeatureExtractor(self)
        self.Serial_feature_extractor: Graph.SerialFeatureExtractor = Graph.SerialFeatureExtractor(self)

        self.graph_duration = self._init_graph_duration()

    def _init_graph_duration(self) -> float:
        attrs = ["forward_times", "backward_times", "optimizer_times"]
        min_, max_ = np.Inf, 0
        for node in self.nodes:
            for attr in attrs:
                t = getattr(node, attr)
                min_ = np.min((min_, *t))
                max_ = np.max((max_, *t))
        graph_duration = max_ - min_
        return graph_duration

    @staticmethod
    def from_data(data: Optional[Dict] = None, dummy: bool = False) -> 'Graph':
        def generate_dummy():
            random_graph_id = ''.join(random.choices(
                string.ascii_letters + string.digits, k=10))
            gpu_type = GPUType.RTX2080Ti
            batch_size = 64
            optimizer_type = OptimizerType.Adam
            operator_types = [OperatorType.Add,
                              OperatorType.Conv2d, OperatorType.Relu]
            args = {
                'input_tensor_size': random.randint(1, 10),
                'weight_tensor_size': random.randint(1, 10),
                'output_tensor_size': random.randint(1, 10),
                'FLOPS': random.uniform(0, 1),
                'hyper_parameters': (random.uniform(0, 1),)
            }

            num_nodes = random.randint(10, 100)
            nodes = list()
            for i in range(num_nodes):
                op_type = random.choice(operator_types)
                op = Operator(op_type, **args)
                last_node = None if len(nodes) == 0 else nodes[-1]

                def added_times(last_node_, attr):
                    if last_node_ is None:
                        start_time = random.uniform(0, 1)
                        duration = random.uniform(0, 1)
                        return start_time, start_time + duration

                    times = getattr(last_node_, attr)
                    start_time = times[-1] + random.uniform(0, 1)
                    duration = random.uniform(0, 1)
                    return start_time, start_time + duration

                current_node = GraphNode(i,
                                         op,
                                         forward_times=added_times(last_node, "forward_times"),
                                         backward_times=added_times(last_node, "backward_times"),
                                         optimizer_times=added_times(last_node, "optimizer_times"), )
                if last_node is not None:
                    last_node.add_neighbors(current_node)
                nodes.append(current_node)
            root_node = nodes[0]
            return Graph(random_graph_id, gpu_type, batch_size, optimizer_type, nodes, root_node)

        if dummy:
            return generate_dummy()
        raise NotImplementedError()

    def subgraphs(self, subgraph_count: Optional[int] = None, subgraph_node_size: Optional[int] = None) -> \
            Tuple[List[List[GraphNode]], Dict[int, int]]:
        # subgraphs, node graph mapping
        if subgraph_count is None and subgraph_node_size is None:
            raise ValueError("Invalid subgraph_count and subgraph_node_size, cannot be None simultaneously.")
        if subgraph_count is not None:
            subgraph_node_size = len(self.nodes) // subgraph_count
        subgraphs = list()
        node_id_to_group_idx = dict()
        for i in range(subgraph_node_size):
            subgraph_nodes = self.nodes[i * subgraph_node_size: (i + 1) * subgraph_node_size]
            subgraphs.append(subgraph_nodes)
            for node in subgraph_nodes:
                node_id_to_group_idx[node.node_id] = i
        return subgraphs, node_id_to_group_idx

    @staticmethod
    def _subgraph_label(subgraph: List[GraphNode]) -> Dict:
        attrs = ["forward_times", "backward_times", "optimizer_times"]
        subgraph_durations = DurationTuple(*(
            abs(getattr(subgraph[0], attr)[0] - getattr(subgraph[-1], attr)[1])
            for attr in attrs
        ))
        node_durations = list()
        for node in subgraph:
            node_duration_label = DurationTuple(*(
                abs(getattr(node, attr)[0] - getattr(node, attr)[1])
                for attr in attrs
            ))
            node_durations.append(node_duration_label)
        return {
            FeatureKeys.Y_SUBGRAPH_FEAT: subgraph_durations,
            FeatureKeys.Y_OP_FEAT: node_durations
        }

    def _optimizer_feature(self):
        return self.optimizer_type.encode()

    class FeatureExtractor(ABC):
        def __init__(self, graph: 'Graph'):
            self.graph: Graph = graph

        def _add_ID(self, x, y, idx):
            x[FeatureKeys.X_ID] = idx
            x[FeatureKeys.X_GRAPH_ID] = self.graph.ID
            y[FeatureKeys.Y_ID] = idx
            y[FeatureKeys.Y_GRAPH_ID] = self.graph.ID

        def node_features(self, op_type_encoding="one-hot", mode="complex", encode_hyper_to_node: bool = True) -> Tuple[
            List[Dict], List[Dict]]:
            graph = self.graph
            X, Y = list(), list()
            optimizer_feature = graph._optimizer_feature()
            for i, node in enumerate(graph.nodes):
                x_op_feature = node.op.to_feature_array(op_type_encoding, mode)
                if encode_hyper_to_node:
                    x_op_feature = np.append(x_op_feature, optimizer_feature)
                x = {
                    FeatureKeys.X_OP_FEAT: x_op_feature
                }
                y = graph._subgraph_label([node])
                self._add_ID(x, y, i)
                X.append(x)
                Y.append(y)
            return X, Y

        def subgraph_features(self, subgraph_node_size: int) -> Tuple[List[Dict], List[Dict]]:
            graph = self.graph
            subgraphs, _ = graph.subgraphs(subgraph_node_size=subgraph_node_size)
            X, Y = list(), list()

            for i, subgraph in enumerate(subgraphs):
                x, y = self._subgraph_feature(subgraph)
                self._add_ID(x, y, i)
                X.append(x)
                Y.append(y)
            return X, Y

        def full_graph_feature(self, *args) -> Tuple[Dict, Dict]:
            X, Y = self.subgraph_features(len(self.graph.nodes))
            return X[0], Y[0]

        @abstractmethod
        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[Dict, Dict]:
            pass

    class GNNBasedFeatureExtractor(FeatureExtractor):
        def __init__(self, graph: 'Graph'):
            super().__init__(graph)

        def _postprocess_matrix(self, feature_matrix, adjacency_matrix) -> Tuple[np.ndarray, np.ndarray]:
            optimizer_node_feature = self.graph._optimizer_feature()
            feature_matrix.append(optimizer_node_feature)
            adjacency_matrix.append(np.zeros(len(adjacency_matrix[0])))

            feature_matrix = pad_np_vectors(feature_matrix)

            feature_matrix = np.array(feature_matrix)
            adjacency_matrix = np.array(adjacency_matrix)
            return feature_matrix, adjacency_matrix

        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[Dict[str, np.ndarray], Dict]:
            feature_matrix = list()
            for node in nodes:
                feature_matrix.append(node.op.to_feature_array(mode="complex"))

            adjacency_matrix = list()
            for node in nodes:
                vector = np.zeros(len(nodes) + 1)
                for neighbor in node.neighbors:
                    vector[neighbor.node_id] = 1

            feature_matrix, adjacency_matrix = self._postprocess_matrix(feature_matrix, adjacency_matrix)
            labels = self.graph._subgraph_label(nodes)
            return {
                FeatureKeys.X_SUBGRAPH_FEAT: feature_matrix,
                FeatureKeys.X_SUBGRAPH_ADJACENCY_FEAT: adjacency_matrix
            }, labels

        def full_graph_feature(self, subgraph_count: int) -> Tuple[Dict[str, np.ndarray], Dict]:
            graph = self.graph
            subgraph_node_size = len(graph.nodes) // subgraph_count
            subgraphs = list()
            node_id_to_group_idx = dict()
            for i in range(subgraph_count):
                subgraph_nodes = graph.nodes[i * subgraph_node_size: (i + 1) * subgraph_node_size]
                subgraphs.append(subgraph_nodes)
                for node in subgraph_nodes:
                    node_id_to_group_idx[node.node_id] = i

            feature_matrix = list()
            for subgraph in subgraphs:
                subgraph_features = list()
                for node in subgraph:
                    node_feature = np.array(node.op.to_feature_array(mode="complex"))
                    subgraph_features.append(node_feature)
                subgraph_features = pad_np_vectors(subgraph_features)
                feature = np.sum(subgraph_features)
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

            feature_matrix, adjacency_matrix = self._postprocess_matrix(feature_matrix, adjacency_matrix)
            labels = Graph._subgraph_label(graph.nodes)
            return {
                FeatureKeys.X_SUBGRAPH_FEAT: feature_matrix,
                FeatureKeys.X_SUBGRAPH_ADJACENCY_FEAT: adjacency_matrix
            }, labels

    class SerialFeatureExtractor(FeatureExtractor):
        def __init__(self, graph: 'Graph'):
            super().__init__(graph)

        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
            feature_matrix = list()
            for node in nodes:
                feature = node.op.to_feature_array(mode="complex")
                feature = np.array(feature)
                feature_matrix.append(feature)

            optimizer_node_feature = self.graph._optimizer_feature()
            feature_matrix.append(optimizer_node_feature)
            feature_matrix = pad_np_vectors(feature_matrix)
            feature_matrix = np.array(feature_matrix)
            labels = self.graph._subgraph_label(nodes)
            return {
                FeatureKeys.X_SUBGRAPH_FEAT: feature_matrix
            }, labels
