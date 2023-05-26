import json
import logging
import os
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Mapping, List

import numpy as np
import torch.optim
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler, ConstantLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import autocast
from config import TrainConfig, EvalConfig
from data.dataset import MDataset, DatasetFactory, Graph
from objects import ModelType
from trainers import MModule
from .util import nested_detach
ckpts_dir = pathlib.Path(__file__).parent.parent / 'ckpts'
logs_dir = pathlib.Path(__file__).parent.parent / 'logs'


class Executor(ABC):
    def __init__(self, conf: TrainConfig|EvalConfig):
        self.model_type: ModelType = self._init_model_type()
        if isinstance(conf, TrainConfig):
            self._init_save_path()
        self.conf: TrainConfig|EvalConfig = conf
        self.train_graphs = DatasetFactory.load_graphs(self.conf.dataset_environment,
                                                       train_or_val="train",
                                                       dummy=self.conf.dataset_dummy)
        self.eval_graphs = DatasetFactory.load_graphs(self.conf.dataset_environment,
                                                      train_or_val="val",
                                                      dummy=self.conf.dataset_dummy)

        self.train_ds: MDataset|None = None
        self.eval_ds: MDataset|None = None
        self.train_records: Dict = dict()

    @abstractmethod
    def _init_model_type(self) -> ModelType:
        pass

    def _init_save_path(self):
        time_format = "%Y-%m-%d_%H-%M-%S"
        time_str = time.strftime(time_format)
        self.save_path = str(ckpts_dir / self.model_type.name / time_str)
        try:
            os.makedirs(self.save_path)
        except IOError:
            logging.fatal("Cannot create save path: %s" % self.save_path)
            exit(-1)

    def load_dataset(self, mode="train") -> MDataset:
        if mode == "train" and self.train_ds is not None:
            return self.train_ds
        if mode == "eval" and self.eval_ds is not None:
            return self.eval_ds
        return self._load_dataset(mode=mode)

    @abstractmethod
    def _load_dataset(self, mode="train") -> MDataset:
        pass

    def init_model(self, processed_train_ds: MDataset, resume_from_ckpt) -> MModule | Any:
        if resume_from_ckpt is not None:
            model = self._load_ckpt(resume_from_ckpt)
            return model
        return self._init_model(processed_train_ds=processed_train_ds)

    @abstractmethod
    def _init_model(self, processed_train_ds: MDataset) -> MModule | Any:
        pass

    @staticmethod
    def _load_ckpt(resume_from_ckpt) -> MModule | Any:
        ckpt_filepath = str(pathlib.Path(ckpts_dir) / resume_from_ckpt)
        model = torch.load(ckpt_filepath)
        return model

    def train(self):
        train_ds = self.load_dataset(mode="train")
        eval_ds = self.load_dataset(mode="eval")
        processed_train_ds = self.preprocess_dataset(train_ds)
        processed_eval_ds = self.preprocess_dataset(eval_ds)
        self._train(processed_train_ds, processed_eval_ds)

    def _create_optimizer_and_scheduler(self, model: Module, num_training_steps) -> Tuple[
        torch.optim.Optimizer, LRScheduler]:
        optimizer_cls = self.conf.optimizer_cls
        lr = self.conf.learning_rate
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        lr_scheduler = ConstantLR(optimizer=optimizer)
        return optimizer, lr_scheduler

    def _train(self, processed_train_ds: MDataset, processed_eval_ds: MDataset):
        train_dl = DataLoader(processed_train_ds, batch_size=self.conf.batch_size, shuffle=True)
        model = self.init_model(processed_train_ds, self.conf.resume_from_ckpt)
        model.train()
        curr_train_step = 0
        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(model, len(train_dl))
        for epoch in range(self.conf.num_train_epochs):
            logging.info(f"{self.model_type} training epoch %d" % epoch)
            for i, data in enumerate(tqdm(train_dl)):
                optimizer.zero_grad()
                features, labels = data
                outputs = model(features)
                loss = model.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                curr_train_step += 1
                loss_value = float(nested_detach(loss))
                self.train_records.setdefault("loss", list())
                self.train_records["loss"].append(loss_value)
                if curr_train_step % self.conf.eval_steps == 0:
                    logging.info(f"{self.model_type} eval at step {curr_train_step}")
                    model.eval()
                    metrics = self._evaluate(model)
                    logging.info(f"{self.model_type} train loss: {loss_value}, eval metrics: {metrics}")
                    self.train_records.setdefault("eval_metrics", list())
                    self.train_records["eval_metrics"].append({
                        "metrics": metrics,
                        "step": curr_train_step
                    })
                    self.save_model(model, curr_steps=curr_train_step, curr_loss_value=loss_value)
                    model.train()
            lr_scheduler.step()

    @abstractmethod
    def preprocess_dataset(self, ds: MDataset) -> MDataset:
        pass

    def evaluate(self):
        train_ds = self.load_dataset(mode="train")
        processed_train_ds = self.preprocess_dataset(train_ds)
        model = self.init_model(processed_train_ds, self.conf.resume_from_ckpt)
        metrics = self._evaluate(model)
        logging.info(f"{self.model_type} evaluated metrics: {metrics}")
        return metrics

    @abstractmethod
    def _evaluate(self, model) -> Dict[str, float]:
        pass

    def save_model(self, model, curr_steps: int, curr_loss_value: float):
        d = {
            "train_config": self.conf.to_dict(),
            "train_records": self.train_records
        }
        with open(pathlib.Path(self.save_path, "train_records.json"), "w") as f:
            json.dump(d, f, indent="\t")
        self._save_ckpt_to(model, pathlib.Path(self.save_path, f"ckpt_{curr_steps}_{curr_loss_value}.pth"))

    @staticmethod
    def _save_ckpt_to(model, filepath):
        torch.save(model, filepath)
