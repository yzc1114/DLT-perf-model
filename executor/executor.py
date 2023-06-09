import json
import logging
import os
import pathlib
import random
import time
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler, ConstantLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import MDataset, load_graphs
from objects import ModelType, ckpts_dir
from .base_module import MModule
from .util import nested_detach


class Executor(ABC):
    def __init__(self, conf: Config):
        self.model_type: ModelType = self._init_model_type()
        self.model_ckpts_dir = ckpts_dir / self.model_type.name
        if isinstance(conf, Config):
            self._init_save_path()
        self.conf: Config = conf
        self.train_graphs = load_graphs(self.conf.dataset_environment,
                                        dummy=self.conf.dataset_dummy)
        self.eval_graphs = load_graphs(self.conf.dataset_environment,
                                       dummy=self.conf.dataset_dummy)
        self.set_seed()

        self.train_ds: MDataset | None = None
        self.eval_ds: MDataset | None = None
        self.preprocessed_train_ds: MDataset | None = None
        self.preprocessed_eval_ds: MDataset | None = None

        self.train_ds = self._init_dataset(mode="train")
        self.eval_ds = self._init_dataset(mode="eval")
        self.preprocessed_train_ds = self._init_preprocessed_dataset(mode="train")
        self.preprocessed_eval_ds = self._init_preprocessed_dataset(mode="eval")

        self.train_records: Dict = dict()
        self._check_params()

    @staticmethod
    @abstractmethod
    def default_model_params() -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def grid_search_model_params() -> Dict[str, List]:
        pass

    @staticmethod
    def grid_search_transfer_params() -> Dict[str, List] | None:
        return None

    def set_seed(self):
        seed = self.conf.all_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # When running on the CuDNN backend, two further options must be set
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        logging.info(f"Random seed set as {seed}")

    def _check_params(self):
        pass

    @abstractmethod
    def _init_model_type(self) -> ModelType:
        pass

    def _init_save_path(self):
        time_format = "%Y-%m-%d_%H-%M-%S"
        time_str = time.strftime(time_format)
        self.save_path = str(self.model_ckpts_dir / time_str)

    def _ensure_save_dir(self):
        p = pathlib.Path(self.save_path)
        if p.exists():
            assert p.is_dir()
            return
        try:
            os.makedirs(self.save_path)
        except IOError:
            logging.fatal("Cannot create save path: %s" % self.save_path)
            exit(-1)

    @abstractmethod
    def _init_dataset(self, mode="train") -> MDataset:
        pass

    def init_model(self) -> MModule | Any:
        if self.conf.resume_from_ckpt is not None:
            ckpt_filepath = pathlib.Path(self.model_ckpts_dir, self.conf.resume_from_ckpt)
            model = self._load_ckpt(ckpt_filepath)
            return model
        return self._init_model()

    @abstractmethod
    def _init_model(self) -> MModule | Any:
        pass

    @staticmethod
    def _load_ckpt(ckpt_filepath) -> MModule | Any:
        model = torch.load(ckpt_filepath)
        return model

    def _create_optimizer_and_scheduler(self, model: Module, num_training_steps) -> Tuple[
        torch.optim.Optimizer, LRScheduler]:
        optimizer_cls = self.conf.optimizer_cls
        lr = self.conf.learning_rate
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        lr_scheduler = ConstantLR(optimizer=optimizer)
        return optimizer, lr_scheduler

    def train(self):
        processed_train_ds = self.preprocessed_train_ds
        train_dl = DataLoader(processed_train_ds, batch_size=self.conf.batch_size, shuffle=True)
        model = self.init_model()
        if self.conf.transfer_params is not None:
            model.prepare_transfer(**self.conf.transfer_params)
        model.train()
        curr_train_step = 0
        optimizer, lr_scheduler = self._create_optimizer_and_scheduler(model, len(train_dl))
        start = time.time_ns()
        logging.info(f"{self.model_type} start training.")
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
                    now = time.time_ns()
                    train_dur = (now - start) / 1e9
                    logging.info(f"{self.model_type} trained for {train_dur} seconds.")
                    logging.info(f"{self.model_type} eval at step {curr_train_step}.")
                    model.eval()
                    metrics = self._evaluate(model)
                    logging.info(f"{self.model_type} train loss: {loss_value}, eval metrics: {metrics}")
                    metrics["train_loss"] = loss_value

                    self.train_records.setdefault("eval_metrics", list())
                    self.train_records["eval_metrics"].append({
                        "metrics": metrics,
                        "step": curr_train_step,
                        "duration": train_dur
                    })
                    self.save_model(model, curr_steps=curr_train_step, curr_loss_value=loss_value)
                    model.train()
            lr_scheduler.step()
        self.save_train_plot()

    def save_train_plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        eval_metrics = self.train_records["eval_metrics"]

        def get_list(metric_key):
            l = list()
            for eval_metric in eval_metrics:
                l.append(eval_metric["metrics"][metric_key])
            return l

        x_step = self.conf.eval_steps
        X = [x_step * i for i in range(len(eval_metrics))]
        # train loss, eval loss
        line_plots = (
            ["train_loss", "eval_loss"],
            ["MRE"],
            ["RMSE"]
        )
        for i, line_plot_keys in enumerate(line_plots):
            ax = axes[i]
            for key in line_plot_keys:
                ax.plot(X, get_list(key), label=key)
            ax.set_xlabel("steps")
            ax.legend()
        fig_save_path = str(pathlib.Path(self.save_path, "train_plot.png"))
        fig.savefig(fig_save_path)

    def _init_preprocessed_dataset(self, mode="train") -> MDataset:
        mode_to_attr = {
            "train": "preprocessed_train_ds",
            "eval": "preprocessed_eval_ds"
        }
        cache = self.__getattribute__(mode_to_attr[mode])
        if self.__getattribute__(mode_to_attr[mode]) is not None:
            return cache

        ds = self.train_ds if mode == "train" else self.eval_ds
        preprocessed = self._preprocess_dataset(ds)
        return preprocessed

    @abstractmethod
    def _preprocess_dataset(self, ds: MDataset) -> MDataset:
        pass

    def evaluate(self):
        model = self.init_model()
        metrics = self._evaluate(model)
        logging.info(f"{self.model_type} evaluated metrics: {metrics}")
        return metrics

    @abstractmethod
    def _evaluate(self, model) -> Dict[str, float]:
        pass

    def _dl_evaluate_pred(self, model: MModule):
        processed_eval_ds = self.preprocessed_eval_ds
        dl = DataLoader(processed_eval_ds, batch_size=self.conf.batch_size, shuffle=False)
        input_batches = list()
        output_batches = list()
        eval_losses = list()
        for data in dl:
            features, labels = data
            with torch.no_grad():
                outputs = model(features)
            loss = model.compute_loss(outputs, labels)
            eval_loss = float(nested_detach(loss))
            eval_losses.append(eval_loss)
            input_batches.append(features)
            output_batches.append(outputs)
        eval_loss = np.mean(eval_losses)

        return input_batches, output_batches, eval_loss

    def save_model(self, model, curr_steps: int, curr_loss_value: float):
        d = {
            "train_config": self.conf.to_dict(),
            "train_records": self.train_records
        }
        self._ensure_save_dir()
        with open(pathlib.Path(self.save_path, "train_records.json"), "w") as f:
            json.dump(d, f, indent="\t")
        self._save_ckpt_to(model, pathlib.Path(self.save_path, f"ckpt_{curr_steps}.pth"))

    @staticmethod
    def _save_ckpt_to(model, filepath):
        torch.save(model, filepath)
