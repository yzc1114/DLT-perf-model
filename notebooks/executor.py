import json
import logging
import pathlib
import time
from typing import Tuple, Any, Dict, List, Callable

import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import MDataset
from objects import ModelType, ckpts_dir
from base_module import MModule
from base_module import nested_detach


def single_train_loop(model_type: ModelType,
                      conf: Config,
                      preprocessed_train_ds: MDataset,
                      preprocessed_eval_ds: MDataset,
                      model: MModule,
                      compute_eval_metrics: Callable[[List, List, float], Dict],
                      to_device: Callable[[Any, Any], Tuple[Any, Any]],
                      ):
    model_ckpts_dir = ckpts_dir / model_type.name
    save_path = generate_save_path(prefix="single_train", model_ckpts_dir=model_ckpts_dir)
    processed_train_ds = preprocessed_train_ds
    train_records: Dict = dict()
    train_dl = DataLoader(processed_train_ds, batch_size=conf.batch_size, shuffle=True)
    model.to(conf.device)
    if conf.transfer_params is not None:
        model.prepare_transfer(**conf.transfer_params)
    model.train()
    curr_train_step = -1
    optimizer_cls = conf.optimizer_cls
    lr = conf.learning_rate
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    start = time.time_ns()
    logging.info(f"{model_type} start single training.")
    for epoch in range(conf.num_train_epochs):
        logging.info(f"{model_type} training epoch %d" % epoch)
        for i, data in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            features, labels = data
            features, labels = to_device(conf, features, labels)
            outputs = model(features)
            loss = model.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            curr_train_step += 1
            loss_value = float(nested_detach(loss))
            train_records.setdefault("loss", list())
            train_records["loss"].append(loss_value)
            if curr_train_step % conf.eval_steps == 0:
                now = time.time_ns()
                train_dur = (now - start) / 1e9
                logging.info(f"{model_type} trained for {train_dur} seconds.")
                logging.info(f"{model_type} eval at step {curr_train_step}.")
                model.eval()
                input_batches, output_batches, eval_loss = evaluate_pred(conf, model, preprocessed_eval_ds, to_device)
                metrics = compute_eval_metrics(input_batches, output_batches, eval_loss)
                logging.info(f"{model_type} train loss: {loss_value}, eval metrics: {metrics}")
                metrics["train_loss"] = loss_value

                train_records.setdefault("eval_metrics", list())
                train_records["eval_metrics"].append({
                    "metrics": metrics,
                    "step": curr_train_step,
                    "duration": train_dur
                })
                save_model(conf=conf,
                           train_records=train_records,
                           save_path=save_path,
                           model=model,
                           curr_steps=curr_train_step,
                           curr_loss_value=loss_value)
                model.train()
    save_train_plot(conf, train_records, save_path)

def _ensure_save_dir(save_path):
    p = pathlib.Path(save_path)
    if p.exists():
        assert p.is_dir()
        return
    try:
        os.makedirs(save_path)
    except IOError:
        logging.fatal("Cannot create save path: %s" % save_path)
        exit(-1)

def save_model(conf, train_records, save_path, model, curr_steps: int, curr_loss_value: float):
    d = {
        "train_config": conf.raw_config,
        "train_records": train_records
    }
    _ensure_save_dir(save_path=save_path)
    with open(pathlib.Path(save_path, "train_records.json"), "w") as f:
        json.dump(d, f, indent="\t")
    torch.save(model, pathlib.Path(save_path, f"ckpt_{curr_steps}.pth"))


def generate_save_path(model_ckpts_dir: str, prefix: str = "") -> str:
        time_format = "%Y-%m-%d_%H-%M-%S"
        time_str = time.strftime(time_format)
        save_path_name = f"{prefix}{time_str}"
        save_path = f"{str(pathlib.Path(model_ckpts_dir) / save_path_name)}"
        return save_path

def evaluate_pred(conf: Config, model: MModule, ds: MDataset, to_device: Callable[[Config, Any, Any], Tuple[Any, Any]]):
    processed_eval_ds = ds
    dl = DataLoader(processed_eval_ds, batch_size=conf.batch_size, shuffle=False)
    input_batches = list()
    output_batches = list()
    eval_losses = list()
    for data in dl:
        features, labels = data
        features, labels = to_device(conf, features, labels)
        with torch.no_grad():
            outputs = model(features)
        loss = model.compute_loss(outputs, labels)
        eval_loss = float(nested_detach(loss))
        eval_losses.append(eval_loss)
        input_batches.append(features)
        output_batches.append(outputs)
    eval_loss = np.mean(eval_losses)

    return input_batches, output_batches, eval_loss

def save_train_plot(conf, train_records, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # plot loss
    ax = axes[0]
    # todo, 采样
    ax.plot([ i for i in range(len(train_records['loss']))],train_records["loss"], label="loss")
    ax.set_xlabel("steps")
    ax.legend()

    eval_metrics = train_records["eval_metrics"]

    def get_list(metric_key):
        l = list()
        for eval_metric in eval_metrics:
            l.append(eval_metric["metrics"][metric_key])
        return l

    x_step = conf.eval_steps
    X = [x_step * i for i in range(len(eval_metrics))]
    # train loss, eval loss
    line_plots = (
        ["train_loss", "eval_loss"],
        ["MRE"],
        ["RMSE"]
    )
    for i, line_plot_keys in enumerate(line_plots):
        ax = axes[i+1]
        for key in line_plot_keys:
            ax.plot(X, get_list(key), label=key)
        ax.set_xlabel("steps")
        ax.legend()
    fig_save_path = str(pathlib.Path(save_path, "train_plot.png"))
    fig.savefig(fig_save_path)
