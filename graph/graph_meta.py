import random
import string
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from data_util import pad_np_vectors
from objects import GPUType, OptimizerType
from op import Operator, OperatorType
from collections import namedtuple


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


class Label:
    DurationTuple = namedtuple(typename="DurationTuple", field_names=["forward", "backward", "optimization"])
    def __init__(self, subgraph_durations: DurationTuple, node_durations: List[DurationTuple]):
        self.subgraph_durations: Label.DurationTuple = subgraph_durations
        self.node_durations: List[Label.DurationTuple] = node_durations


class Graph:
    def __init__(self,
                 name: Optional[str],
                 gpu_type: Optional[GPUType],
                 batch_size: Optional[int],
                 optimizer_type: Optional[OptimizerType],
                 nodes: Optional[List[GraphNode]],
                 root_node: Optional[GraphNode]):
        self.name: Optional[str] = name
        self.gpu_type: Optional[GPUType] = gpu_type
        self.batch_size: Optional[int] = batch_size
        self.optimizer_type: Optional[OptimizerType] = optimizer_type
        self.nodes: List[GraphNode] = list() if nodes is None else nodes
        self.root_node: Optional[GraphNode] = root_node

        self.GNN_based_feature_extractor: Graph.GNNBasedFeatureExtractor = Graph.GNNBasedFeatureExtractor(self)
        self.Serial_feature_extractor: Graph.SerialFeatureExtractor = Graph.SerialFeatureExtractor(self)

    @staticmethod
    def from_data(data: Optional[Dict]=None, dummy: bool = False) -> 'Graph':
        def generate_dummy():
            random_graph_name = ''.join(random.choices(
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
                def added_times(times):
                    start_time = random.uniform(0, 1) if last_node is None else times[-1] + random.uniform(0,1)
                    duration = random.uniform(0, 1)
                    return start_time, start_time + duration
                current_node = GraphNode(i,
                                         op,
                                         forward_times=added_times(last_node.forward_times),
                                         backward_times=added_times(last_node.backward_times),
                                         optimizer_times=added_times(last_node.optimizer_times),)
                if last_node is not None:
                    last_node.add_neighbors(current_node)
                nodes.append(current_node)
            root_node = nodes[0]
            return Graph(random_graph_name, gpu_type, batch_size, optimizer_type, nodes, root_node)

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
    def _subgraph_label(subgraph: List[GraphNode]) -> Label:
        attrs = ["forward_times", "backward_times", "optimizer_times"]
        subgraph_durations = Label.DurationTuple(*(
            abs(getattr(subgraph[0], attr)[0] - getattr(subgraph[1], attr)[1])
            for attr in attrs
        ))
        node_durations = list()
        for node in subgraph:
            node_duration_label = Label.DurationTuple(*(
                abs(getattr(node, attr)[0] - getattr(node, attr)[1])
                for attr in attrs
            ))
            node_durations.append(node_duration_label)
        return Label(subgraph_durations, node_durations)

    def _optimizer_feature(self):
        return self.optimizer_type.encode()

    class FeatureExtractor(ABC):
        def __init__(self, graph: 'Graph'):
            self.graph: Graph = graph

        def node_features(self, op_type_encoding="one-hot", mode="complex", encode_hyper_to_node: bool=True) -> Tuple[List[Any], List[Label]]:
            graph = self.graph
            X, Y = list(), list()
            optimizer_feature = graph._optimizer_feature()
            for node in graph.nodes:
                x = node.op.to_feature(op_type_encoding, mode)
                if encode_hyper_to_node:
                    x = np.append(x, optimizer_feature)
                y = graph._subgraph_label([node])
                X.append(x)
                Y.append(y)
            return X, Y

        def subgraph_features(self, subgraph_node_size: int) -> Tuple[List[Any], List[Label]]:
            graph = self.graph
            subgraphs, _ = graph.subgraphs(subgraph_node_size=subgraph_node_size)
            X, Y = list(), list()
            for subgraph in subgraphs:
                x, y = self._subgraph_feature(subgraph)
                X.append(x)
                Y.append(y)
            return X, Y

        def full_graph_feature(self, *args) -> Tuple[Any, Label]:
            X, Y = self.subgraph_features(len(self.graph.nodes))
            return X[0], Y[0]

        @abstractmethod
        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[Tuple[np.array, np.array], Label]:
            pass

    class GNNBasedFeatureExtractor(FeatureExtractor):
        def __init__(self, graph: 'Graph'):
            super().__init__(graph)

        def _postprocess_matrix(self, feature_matrix, adjacency_matrix) -> Tuple[np.array, np.array]:
            optimizer_node_feature = self.graph._optimizer_feature()
            feature_matrix.append(optimizer_node_feature)
            adjacency_matrix.append(np.zeros(len(adjacency_matrix[0])))

            feature_matrix = pad_np_vectors(feature_matrix)

            feature_matrix = np.array(feature_matrix)
            adjacency_matrix = np.array(adjacency_matrix)
            return feature_matrix, adjacency_matrix

        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[Tuple[np.array, np.array], Label]:
            feature_matrix = list()
            for node in nodes:
                feature_matrix.append(node.op.to_feature(mode="simple"))

            adjacency_matrix = list()
            for node in nodes:
                vector = np.zeros(len(nodes) + 1)
                for neighbor in node.neighbors:
                    vector[neighbor.node_id] = 1

            feature_matrix, adjacency_matrix = self._postprocess_matrix(feature_matrix, adjacency_matrix)
            labels = self.graph._subgraph_label(nodes)
            return (feature_matrix, adjacency_matrix), labels

        def full_graph_feature(self, subgraph_count: int) -> Tuple[Tuple[np.array, np.array], Label]:
            graph = self.graph
            subgraph_node_size = len(graph.nodes) // subgraph_count
            subgraphs = list()
            node_id_to_group_idx = dict()
            for i in range(subgraph_node_size):
                subgraph_nodes = graph.nodes[i * subgraph_node_size: (i + 1) * subgraph_node_size]
                subgraphs.append(subgraph_nodes)
                for node in subgraph_nodes:
                    node_id_to_group_idx[node.node_id] = i

            feature_matrix = list()
            for subgraph in subgraphs:
                subgraph_features = list()
                for node in subgraph:
                    node_feature = np.array(node.op.to_feature(mode="simple"))
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
            x = feature_matrix, adjacency_matrix
            y = Graph._subgraph_label(graph.nodes)
            return x, y

    class SerialFeatureExtractor(FeatureExtractor):
        def __init__(self, graph: 'Graph'):
            super().__init__(graph)

        def _subgraph_feature(self, nodes: List[GraphNode]) -> Tuple[np.array, Label]:
            feature_matrix = list()
            for node in nodes:
                feature = node.op.to_feature(mode="complex")
                feature = np.array(feature)
                feature_matrix.append(feature)

            optimizer_node_feature = self.graph._optimizer_feature()
            feature_matrix.append(optimizer_node_feature)
            feature_matrix = pad_np_vectors(feature_matrix)
            feature_matrix = np.array(feature_matrix)
            labels = self.graph._subgraph_label(nodes)
            return feature_matrix, labels


