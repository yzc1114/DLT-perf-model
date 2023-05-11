import logging
import pathlib
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Mapping
from typing import Optional, List, Dict, Callable
from typing import Tuple, Any, Union

import numpy as np
import torch
import torch.nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer import EvalPrediction, get_last_checkpoint

from data import DatasetFactory
from data import FeatureKeys
from data import Graph
from data.dataset import MDataset


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


class FullGraphMetricMixin:

    @staticmethod
    def _op_based_full_graph_metrics(inputs_batches: List[Dict], outputs_batches: List, graphs: List[Graph]) -> Dict:
        assert len(inputs_batches) == len(outputs_batches)
        batches_len = len(inputs_batches)

        def compute_op_durations(outputs_):
            logits = outputs_[FeatureKeys.Y_OP_FEAT]
            duration_dim = (0, 3)
            durations = logits[:, duration_dim[0]:duration_dim[1]].sum(dim=1)
            return durations

        graph_id_to_duration_pred = defaultdict(int)
        for idx in range(batches_len):
            inputs = inputs_batches[idx]
            outputs = outputs_batches[idx]
            graph_ids = inputs[FeatureKeys.X_GRAPH_ID]
            op_durations = compute_op_durations(outputs)
            for i, graph_id in enumerate(graph_ids):
                op_duration = op_durations[i].item()
                graph_id_to_duration_pred[graph_id] += op_duration
        duration_metrics = MetricUtil.compute_duration_metrics(graphs, graph_id_to_duration_pred)
        return {
            **duration_metrics
        }


class LossUtilMixin:
    @staticmethod
    def op_based_loss(model, loss_fn, inputs):
        labels = inputs["labels"]
        # here, subgraph equals to op since a subgraph only contains one op
        y_op_features = labels[FeatureKeys.Y_SUBGRAPH_FEAT]
        x_op_features = inputs[FeatureKeys.X_OP_FEAT]
        logits = model(x_op_features)
        outputs = {
            FeatureKeys.Y_OP_FEAT: logits
        }
        loss = loss_fn(outputs[FeatureKeys.Y_OP_FEAT], y_op_features)
        return loss, outputs


class MModule(torch.nn.Module, MetricUtil, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                 resume_from_ckpt: Union[bool, str] | None,
                 use_hugging_face: bool = True,
                 **hgf_kwargs):
        self.resume_from_ckpt: Union[bool, str] | None = resume_from_ckpt
        if use_hugging_face:
            self.trainer = MHuggingFaceTrainer(model_init=model_init,
                                               args=args,
                                               train_dataset=train_dataset,
                                               eval_dataset=eval_dataset,
                                               optimizer_cls=optimizer_cls,
                                               resume_from_ckpt=resume_from_ckpt,
                                               **hgf_kwargs)
        else:
            self.model_init = model_init
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.optimizer_cls = optimizer_cls

    def train(self):
        self.trainer.train()

    def evaluate(self) -> Dict[str, float]:
        return self.trainer.evaluate()


class MHuggingFaceTrainer(Trainer):
    def __init__(self,
                 model_init: Callable,
                 args: TrainingArguments,
                 train_dataset: MDataset,
                 eval_dataset: MDataset,
                 optimizer_cls,
                 resume_from_ckpt: Union[bool, str] | None,
                 **hgf_kwargs):

        self._create_optimizer_and_scheduler = hgf_kwargs.get("_create_optimizer_and_scheduler", None)
        if isinstance(resume_from_ckpt, str):
            args.resume_from_checkpoint = str(pathlib.Path(args.output_dir, resume_from_ckpt))
        else:
            args.resume_from_checkpoint = resume_from_ckpt

        super().__init__(model_init=model_init,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=MHuggingFaceTrainer.compute_metrics,
                         data_collator=MDataset.data_collator)
        self.optimizer_cls = optimizer_cls
        if isinstance(args.resume_from_checkpoint, bool):
            args.resume_from_checkpoint = get_last_checkpoint(args.output_dir)

        if isinstance(args.resume_from_checkpoint, str):
            ckpt_path = args.resume_from_checkpoint
            logging.info(f"resume from checkpoint: {ckpt_path}")
            self._load_from_checkpoint(ckpt_path)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model.loss(inputs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps):
        if self._create_optimizer_and_scheduler is not None:
            self.optimizer, self.lr_scheduler = self._create_optimizer_and_scheduler(self, num_training_steps)
            return
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ConstantLR(optimizer=self.optimizer)

    @staticmethod
    def compute_metrics(eval_prediction: EvalPrediction) -> Dict:
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

        logging.info(f"***** Running Evaluation *****")
        logging.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        logging.info(f"  Batch size = {batch_size}")

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
