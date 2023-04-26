import math
import time
from typing import Optional, List, Dict, Callable, Tuple
from typing import Mapping

import numpy as np
import torch
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer import EvalLoopOutput, EvalPrediction, speed_metrics, find_batch_size, \
    nested_concat, nested_numpify, nested_truncate, denumpify_detensorize
from transformers.trainer import logging

from data.dataset import MDataset
from models import MModule

logger = logging.get_logger(__name__)


class MTrainer(Trainer):
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
                         compute_metrics=MTrainer.compute_metrics,
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
        loss = np.mean(loss_batches)
        full_graph_metrics = model.full_graph_metrics(inputs_batches, outputs_batches, eval_dataset)
        metrics = {
            "eval_loss": loss,
            **full_graph_metrics
        }

        self.log(metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        losses = None
        inputs_batches = list()
        outputs_batches = list()
        for step, inputs in enumerate(dataloader):
            loss, outputs, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            losses = loss if losses is None else torch.cat((losses, loss), dim=0)
            inputs_batches.append(inputs)
            outputs_batches.append(outputs)

        eval_dataset = getattr(dataloader, "dataset", None)

        full_graph_metrics = model.full_graph_metrics(inputs_batches, outputs_batches, eval_dataset)
        loss = losses.mean().detach().cpu().numpy()
        metrics = {
            "eval_loss": loss,
            **full_graph_metrics
        }
        num_samples = 64

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

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
