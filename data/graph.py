import math
import random
import string
from collections import namedtuple
from typing import List, Optional, Dict, Tuple

import numpy as np

from objects import GPUType, OptimizerType
from .op import Operator, OperatorType


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

    @staticmethod
    def dummy_node():
        return GraphNode(node_id=random.randint(1000, 1e6), op=Operator.dummy_op(), forward_times=(0, 0), backward_times=(0, 0),
                         optimizer_times=(0, 0))

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
    def from_data(data: Optional[Dict] = None, dummy: bool = False, seed: int = 0) -> 'Graph':
        rand = random.Random(seed)

        def generate_dummy():
            random_graph_id = ''.join(rand.choices(
                string.ascii_letters + string.digits, k=10))
            gpu_type = GPUType.RTX2080Ti
            batch_size = 64
            optimizer_type = OptimizerType.Adam
            operator_types = [OperatorType.Add,
                              OperatorType.Conv2d, OperatorType.Relu]

            num_nodes = rand.randint(10, 100)
            nodes = list()
            for i in range(num_nodes):
                op_type = rand.choice(operator_types)
                hyper_param_cnt = rand.randint(0, 10)
                args = {
                    'input_tensor_size': rand.randint(1, 10),
                    'weight_tensor_size': rand.randint(1, 10),
                    'output_tensor_size': rand.randint(1, 10),
                    'FLOPS': rand.uniform(0, 1),
                    'hyper_parameters': tuple(rand.uniform(0, 1) for i in range(hyper_param_cnt))
                }
                op = Operator(op_type, **args)
                last_node = None if len(nodes) == 0 else nodes[-1]

                def added_times(last_node_, attr):
                    if last_node_ is None:
                        start_time = rand.uniform(0, 1)
                        duration = rand.uniform(0, 1)
                        return start_time, start_time + duration

                    times = getattr(last_node_, attr)
                    start_time = times[-1] + rand.uniform(0, 1)
                    duration = rand.uniform(0, 1)
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

    def subgraphs(self, subgraph_count: Optional[int]=None, subgraph_node_size: Optional[int] = None, step: Optional[int]=None) -> \
            Tuple[List[List[GraphNode]], Dict[int, int]]:
        if subgraph_node_size is None:
            assert subgraph_count is not None
            subgraph_node_size = math.ceil(len(self.nodes) / subgraph_count)
        # subgraphs, node graph mapping
        if step is None:
            step = subgraph_node_size
        subgraphs = list()
        node_id_to_group_idx = dict()
        idx = 0
        while True:
            if idx >= len(self.nodes):
                break
            subgraph_nodes = self.nodes[idx: 
                                        min(idx + subgraph_node_size, len(self.nodes))]
            subgraphs.append(subgraph_nodes)
            for node in subgraph_nodes:
                node_id_to_group_idx[node.node_id] = idx//step
            dummy_node_require = False
            while len(subgraph_nodes) < subgraph_node_size:
                subgraph_nodes.append(GraphNode.dummy_node())
                dummy_node_require = True
            if dummy_node_require:
                break
            idx += step
        return subgraphs, node_id_to_group_idx

    def graph_meta_feature(self):
        return self.optimizer_type.encode()
