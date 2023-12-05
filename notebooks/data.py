import pathlib
import os
import pandas as pds
from pandas import DataFrame
from functools import lru_cache
from typing import List, Dict
from torch.utils.data import Dataset
from objects import Environment
import math
import random
import string
from typing import List, Optional, Dict, Tuple

import numpy as np
import multiprocessing as mp
from enum import Enum
from functools import lru_cache
from typing import Tuple, Optional, Union, List, Dict
from logger import logging

repo_root = pathlib.Path(__file__).parent.parent.absolute()
datasets_path = str(repo_root / "datasets")
configs_path = str(pathlib.Path(__file__).parent / "configs")
print(f"datasets_path: {datasets_path}")
print(f"configs_path: {configs_path}")


op_freq = {}

with open(str(pathlib.Path(configs_path) / "op_frequency.dict"), "r") as f:
    op_freq = eval(f.read())


class OperatorMode(Enum):
    Forward = 0
    Backward = 1
    Update = 2

    @lru_cache(maxsize=None)
    def encode(self) -> List:
        op_modes = [mode for mode in OperatorMode]
        return [1 if self == op_mode_ else 0 for op_mode_ in op_modes]


class Operator:
    def __init__(self,
                 operator_type_id: int,
                 operator_mode: OperatorMode,
                 batch_size: int = 0,
                 FLOPS: int = 0,
                 bytes: int = 0,
                 hyper_parameters: Optional[Tuple[Union[float, int]]] = None
                 ):
        self.operator_type_id: int = operator_type_id
        self.operator_mode: OperatorMode = operator_mode
        self.batch_size: int = batch_size
        self.FLOPS: int = FLOPS
        self.bytes: int = bytes
        self.hyper_parameters: Optional[Tuple[Union[float, int]]
        ] = hyper_parameters

    @staticmethod
    def dummy_op():
        return Operator(0, OperatorMode.Forward)

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id(i: int) -> List:
        l = [0] * 238
        l[i - 1] = 1
        return l

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_static(i: int) -> List:
        return [i]

    @lru_cache(maxsize=None)
    @staticmethod
    def encode_op_type_id_frequency(i: int) -> List:
        return [op_freq[i]]

    def to_feature_array(self, mode="complex"):
        if mode == "complex":
            complex_feature_vector = [
                *Operator.encode_op_type_id_static(self.operator_type_id),
                *self.operator_mode.encode(),
                self.batch_size,
                self.FLOPS,
                self.bytes,
            ]
            if self.hyper_parameters is not None:
                complex_feature_vector.extend(self.hyper_parameters)
            return np.array(complex_feature_vector)
        elif mode == "simple":
            simple_feature_vector = [
                *self.encode_op_type_id_static(self.operator_type_id),
                *self.operator_mode.encode(),
                self.batch_size,
            ]
            return np.array(simple_feature_vector)
        else:
            raise ValueError(
                "Invalid mode. Mode must be 'complex' or 'simple'.")



class GraphNode:
    def __init__(self,
                 node_id: int,
                 op: Operator,
                 duration: int,
                 gap: int):
        self.node_id: int = node_id
        self.op: Operator = op
        self.duration: int = duration
        self.gap: int = gap

    @staticmethod
    def dummy_node():
        return GraphNode(node_id=random.randint(1000, 1e6), op=Operator.dummy_op(), duration=0, gap=0)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id

    def __ne__(self, other):
        return not self.__eq__(other)


class Graph:

    operator_modes_map = {
        1: OperatorMode.Forward,
        2: OperatorMode.Backward,
        3: OperatorMode.Update
    }
    def __init__(self,
                 ID: Optional[str],
                 environment: Optional[Environment],
                 batch_size: Optional[int],
                 nodes: Optional[List[GraphNode]],
                 root_node: Optional[GraphNode]):
        self.ID: Optional[str] = ID
        self.environment: Optional[Environment] = environment
        self.batch_size: Optional[int] = batch_size
        self.nodes: List[GraphNode] = list() if nodes is None else nodes
        self.root_node: Optional[GraphNode] = root_node
        self.graph_duration = self._init_graph_duration()

    def _init_graph_duration(self) -> float:
        graph_duration = 0
        for node in self.nodes:
            graph_duration += node.duration + node.gap
        return graph_duration

    @staticmethod
    def from_data(env: Environment,
                  filename: Optional[str] = None,
                  df: Optional[DataFrame] = None,
                  dummy: bool = False,
                  seed: int = 0) -> 'Graph':
        def generate_dummy():
            rand = random.Random(seed)
            random_graph_id = ''.join(rand.choices(
                string.ascii_letters + string.digits, k=10))
            env = Environment.from_str("RTX2080Ti_CPU100")
            batch_size = 64
            operator_modes = [OperatorMode.Forward,
                              OperatorMode.Backward, OperatorMode.Update]

            num_nodes = rand.randint(10, 100)
            nodes = list()
            for i in range(num_nodes):
                op_type = rand.choice([0, 1, 2, 3])
                op_mode = rand.choice(operator_modes)
                batch_size = rand.choice([32, 64, 128])
                hyper_param_cnt = rand.randint(0, 10)
                args = {
                    'FLOPS': rand.uniform(0, 1),
                    'batch_size': batch_size,
                    'hyper_parameters': tuple(rand.uniform(0, 1) for i in range(hyper_param_cnt))
                }
                op = Operator(op_type, op_mode, **args)
                duration, gap = rand.uniform(0, 1), rand.uniform(0, 1)
                current_node = GraphNode(i,
                                         op,
                                         duration=duration,
                                         gap=gap)
                nodes.append(current_node)
            root_node = nodes[0]
            return Graph(random_graph_id, env, batch_size, nodes, root_node)

        if dummy:
            return generate_dummy()

        # d = df.to_dict()
        total_op = len(df)
        nodes = list()
        for i in range(total_op):
            operator_type_id = int(df.iloc[i]["op"])
            operator_mode = OperatorMode(Graph.operator_modes_map[df.iloc[i]["dir"]])
            # op_hyper_parameters = [] #list(eval(df.iloc[i]["params"]))
            params = df.iloc[i]["params"]
            # params is like a "[1, 2, 3]" string. Turn it into python list. Do not use eval due to security issue.
            op_hyper_parameters = list(map(int, params[1:-1].split(",")))
            # 某些算子的参数超过30个，这里只取前30个
            op_hyper_parameters = op_hyper_parameters[:30]
            if len(op_hyper_parameters) < 30:
                # pad to 30
                op_hyper_parameters.extend([0] * (30 - len(op_hyper_parameters)))

            flops = int(df.iloc[i]["flops"])
            bytes = int(df.iloc[i]["bytes"])
            kduration = int(df.iloc[i]["kduration"]) / 1000.  # us
            space = int(df.iloc[i]["space"]) / 1000.  # us
            batch_size = int(df.iloc[i]["batch"])
            op = Operator(operator_type_id=operator_type_id,
                          operator_mode=operator_mode,
                          FLOPS=flops,
                          bytes=bytes,
                          batch_size=batch_size,
                          hyper_parameters=op_hyper_parameters)
            current_node = GraphNode(i,
                                     op,
                                     duration=kduration,
                                     gap=space)
            nodes.append(current_node)
        root_node = nodes[0]
        return Graph(filename, env, batch_size, nodes, root_node)

    def subgraphs(self, subgraph_count: Optional[int] = None, subgraph_node_size: Optional[int] = None,
                  step: Optional[int] = None) -> \
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
                node_id_to_group_idx[node.node_id] = idx // step
            dummy_node_require = False
            while len(subgraph_nodes) < subgraph_node_size:
                subgraph_nodes.append(GraphNode.dummy_node())
                dummy_node_require = True
            if dummy_node_require:
                break
            idx += step
        return subgraphs, node_id_to_group_idx


class MDataset(Dataset):
    def __init__(self, features: List[Dict], labels: List[Dict]):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

        return x, y


def _load_graphs_of_filenames(inputs: List):
    _graphs = list()
    envs, filenames, csvs = inputs
    for env, filename, csv in zip(envs, filenames, csvs):
        filecsv = csv.loc[csv["filename"] == filename]
        graph = Graph.from_data(env, filename=filename, df=filecsv)
        logging.info(f"Loaded {filename}, {len(filecsv)} rows")
        _graphs.append(graph)
    return _graphs

def load_graphs(environment: Environment, train_or_eval: str = "train", use_dummy: bool = False, max_row: int = 100_000) -> List[Graph]:
    def _load_graphs():
        if use_dummy:
            return list(Graph.from_data(environment, dummy=True, seed=seed) for seed in range(500))
        data_dir = pathlib.Path(datasets_path) / f"{environment}" / train_or_eval
        # Load data from directory
        _graphs = list()
        if os.path.exists(str(data_dir / "merged.csv")):
            # load merged.csv
            logging.info(f"Loading merged.csv")
            csv = pds.read_csv(str(data_dir / "merged.csv"), nrows=max_row)
            csv = csv[csv["seq"] != "-1"]
            logging.info(f"Loaded merged.csv, {len(csv)} rows")
            # list all unique filenames
            filenames = csv["filename"].unique() 
            for filename in filenames:
                filecsv = csv.loc[csv["filename"] == filename]
                graph = Graph.from_data(environment, filename=filename, df=filecsv)
                logging.info(f"Loaded {filename}, {len(filecsv)} rows")
                _graphs.append(graph)
            return _graphs

        # if not merged, load data from each file
        curr_row = 0
        for filename in os.listdir(str(data_dir)):
            if not filename.endswith(".csv"):
                continue
            csv = pds.read_csv(str(data_dir / filename))
            # 删除optimizer的数据
            csv = csv[csv["seq"] != "-1"]
            logging.info(f"Loaded {filename}, {len(csv)} rows")
            graph = Graph.from_data(environment, filename=filename, df=csv)
            _graphs.append(graph)
            curr_row += len(csv)
            if curr_row >= max_row:
                break
        return _graphs

    logging.info(f"Loading graphs {train_or_eval}")
    graphs = _load_graphs()
    return graphs

