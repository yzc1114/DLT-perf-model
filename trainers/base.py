from abc import ABC
from abc import abstractmethod
from typing import Mapping
from typing import Optional, List, Dict, Callable
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer import EvalPrediction
from transformers.trainer import logging

from data import DatasetFactory
from data import Graph
from data.dataset import MDataset

logger = logging.get_logger(__name__)


class MetricUtil:
    @staticmethod
    def compute_duration_metrics(graphs: List[Graph], graph_id_to_duration_pred: Dict[str, float]) -> Dict:
        y_hat, y = list(), list()
        for graph in graphs:
            pred = graph_id_to_duration_pred[graph.ID]
            ground_truth = graph.graph_duration
            y_hat.append(pred)
            y.append(ground_truth)
        y_hat = np.array(y_hat)
        y = np.array(y)
        MRE = np.sum(np.abs(y - y_hat) / y) / len(y)
        RMSE = np.sqrt(np.sum(np.power(y - y_hat, 2)) / len(y))
        return {
            "MRE": MRE,
            "RMSE": RMSE
        }


class MModule(torch.nn.Module, MetricUtil, ABC):
    @abstractmethod
    def loss(self, inputs) -> Tuple[torch.Tensor, Any]:
        pass

    def full_graph_metrics(self, inputs_batches: List[List], outputs_batches: List, eval_dataset: MDataset) -> Dict:
        graphs = DatasetFactory.graphs_cache[eval_dataset.graphs_cache_key]
        return self._full_graph_metrics(inputs_batches, outputs_batches, graphs)

    @abstractmethod
    def _full_graph_metrics(self, inputs_batches, outputs_batches, graphs) -> Dict:
        pass


class MTrainer:
    def __init__(self,
                 model_init: Callable,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls,
                 use_hugging_face: bool = True):
        if use_hugging_face:
            self.trainer = MHuggingFaceTrainer(model_init=model_init,
                                               args=args,
                                               train_dataset=train_dataset,
                                               eval_dataset=eval_dataset,
                                               optimizer_cls=optimizer_cls)
        else:
            self.model_init = model_init
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.optimizer_cls = optimizer_cls

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()


class MHuggingFaceTrainer(Trainer):
    def __init__(self,
                 model_init: Callable,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls):
        super().__init__(model_init=model_init,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=MHuggingFaceTrainer.compute_metrics,
                         data_collator=MDataset.data_collator,
                         )
        self.model_init = model_init
        self.optimizer_cls = optimizer_cls
        self.args = args

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model.loss(inputs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ConstantLR(optimizer=self.optimizer)

    @staticmethod
    def compute_metrics(eval_prediction: EvalPrediction) -> Dict:
        # Callable[[EvalPrediction], Dict]
        # eval_prediction.label_ids
        print(eval_prediction)
        return {}

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if eval_dataset is None:
            eval_dataset = getattr(eval_dataloader, "dataset")

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        loss_batches = list()
        inputs_batches = list()
        outputs_batches = list()
        for step, inputs in enumerate(eval_dataloader):
            loss, outputs, _ = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys)
            loss_batches.append(loss.item())
            inputs_batches.append(inputs)
            outputs_batches.append(outputs)
        loss = float(np.mean(loss_batches))
        full_graph_metrics = model.full_graph_metrics(inputs_batches, outputs_batches, eval_dataset)
        metrics = {
            "eval_loss": loss,
            **full_graph_metrics
        }

        self.log(metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def prediction_step(
            self,
            model: MModule,
            inputs: Dict,
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = model.loss(inputs)
        labels = nested_detach(inputs.get("labels"))
        outputs = nested_detach(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        loss = loss.mean().detach()

        return loss, outputs, labels


def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    if isinstance(tensors, np.ndarray):
        return tensors
    return tensors.detach()
