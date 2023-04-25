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
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        all_preds = None
        all_labels = None
        metrics = {
            "eval_loss": 0
        }
        num_samples = 64

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: MModule,
            inputs: Dict,
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                outputs = model.loss(inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]
        labels = nested_detach(inputs.get("labels"))
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return loss, logits, labels

def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    if isinstance(tensors, np.ndarray):
        return tensors
    return tensors.detach()
