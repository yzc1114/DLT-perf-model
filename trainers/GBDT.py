import os
import pathlib
import pickle
from collections import defaultdict
from typing import List, Dict, Optional, Callable

import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments

from config import TrainConfig
from data import FeatureKeys, MDataset, DatasetFactory
from .base import MTrainer, MetricUtil


def GBDT_init(train_config: TrainConfig, train_ds: MDataset):
    config_params = train_config.model_params
    model_params = {
        "n_estimators": config_params.get("n_estimators", 500),
        "max_depth": config_params.get("max_depth", 4),
        "min_samples_split": config_params.get("min_samples_split", 4),
        "learning_rate": config_params.get("learning_rate", 0.001),
        "loss": "squared_error",
    }
    reg = ensemble.GradientBoostingRegressor(**model_params)
    return reg


class GBDTTrainer(MTrainer):
    def __init__(self,
                 model_init: Callable,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls):
        super().__init__(model_init,
                         args,
                         train_dataset,
                         eval_dataset,
                         optimizer_cls)
        self.reg = None

    @staticmethod
    def dataset_to_samples(dataset: MDataset):
        samples = list(dataset)
        collated = MDataset.data_collator(samples)
        return collated

    @staticmethod
    def _save_filename(loss):
        return "checkpoint_%.2f.pickle".format(loss)

    @staticmethod
    def _loss_from_filename(filename: str) -> float:
        return float(filename[:filename.index(".")].split("_")[-1])

    def train(self, **kwargs):
        dataset = getattr(self.get_train_dataloader(), "dataset")
        assert isinstance(dataset, MDataset)
        samples = GBDTTrainer.dataset_to_samples(dataset)
        X, Y = samples[FeatureKeys.X_OP_FEAT], samples["labels"][FeatureKeys.Y_SUBGRAPH_FEAT]

        self.reg = self.model_init()
        self.reg.fit(X, Y)

        metrics = self.evaluate()
        eval_loss = metrics["eval_loss"]
        with open(pathlib.Path(self.args.output_dir, self._save_filename(eval_loss)), "wb") as f:
            pickle.dump(self.reg, f)

    def _load_best_reg(self):
        filenames = os.listdir(self.args.output_dir)
        min_, best_filename = np.inf, None
        for filename in filenames:
            loss = self._loss_from_filename(filename)
            if loss < min_:
                min_ = loss
                best_filename = filename
        if best_filename is None:
            raise RuntimeError("GBDT trained regression models not exist.")
        with open(pathlib.Path(self.args.output_dir, best_filename), "rb") as f:
            return pickle.load(f)

    def evaluate(
            self,
            eval_dataset: Optional[MDataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = getattr(self.get_eval_dataloader(), "dataset")
        samples = GBDTTrainer.dataset_to_samples(eval_dataset)
        X, Y = samples[FeatureKeys.X_OP_FEAT], samples["labels"][FeatureKeys.Y_SUBGRAPH_FEAT]
        reg = self.reg if self.reg is not None else self._load_best_reg()
        Y_pred = reg.predict(X)
        mse = mean_squared_error(Y, Y_pred)

        assert len(X) == len(Y)
        dataset_len = len(X)

        op_durations = Y[:, :3].sum(dim=1)
        graph_id_to_duration_pred = defaultdict(int)
        for i in range(dataset_len):
            graph_id = samples[FeatureKeys.X_GRAPH_ID][i]
            op_pred = op_durations[i]
            graph_id_to_duration_pred[graph_id] += op_pred

        graphs = DatasetFactory.graphs_cache[eval_dataset.graphs_cache_key]
        duration_metrics = MetricUtil.compute_duration_metrics(graphs, graph_id_to_duration_pred)
        return {
            "eval_loss": mse,
            **duration_metrics
        }
