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
from itertools import product

from config import Config
from data import MDataset
from objects import ModelType, ckpts_dir
from base_module import MModule, nested_detach



def grid_search_loop(model_type: ModelType,
                      conf: Config,
                      preprocessed_train_ds: MDataset,
                      preprocessed_eval_ds: MDataset,
                      compute_eval_metrics: Callable[[List, List, float], Dict],
                      to_device: Callable[[Any, Any], Tuple[Any, Any]],
                      init_model: Callable[[Config], MModule],
                      on_common_params:bool = True,
                      model_specific_gs_params_name: str = 'model_params',
                      ):
    grid_search_params = dict()
    for k, v in conf.to_dict().items():
        if not isinstance(v, list):
            grid_search_params[k] = [v]
        else: 
            grid_search_params[k] = v
        if not on_common_params:
            grid_search_params[k] = [grid_search_params[k][0]]
    
    grid_search_item_lens = list((len(v) for v in grid_search_params.values()))    
    
    model = init_model()
    model_specific_gs_params = getattr(model, f"grid_search_{model_specific_gs_params_name}")()
    for k, v in model_specific_gs_params.items():
        grid_search_item_lens.append(len(v))
    
    total_search_items = np.prod(grid_search_item_lens)

    logging.info(
        f"{model_type} grid search on common params: {on_common_params}, model specific grid search params name: {model_specific_gs_params}.")
    logging.info(f"total search items: {total_search_items}.")

    search_param_keys = sorted(grid_search_params.keys())
    search_items = [grid_search_params[k] for k in search_param_keys]
    print(search_items)
    all_search_confs = []
    all_start_time = time.time()
    for search_item_combination in list(product(*search_items)):
        # main search params. all models share
        curr_conf = dict()
        for i in range(len(search_item_combination)):
            search_param_key = search_param_keys[i]
            search_item = search_item_combination[i]
            curr_conf[search_param_key] = search_item

        # model param grid search params. specialized to each model
        keys = sorted(model_specific_gs_params.keys())
        items = [model_specific_gs_params[k] for k in model_specific_gs_params]
        for model_param_search_item_combination in list(product(*items)):
            curr_conf[model_specific_gs_params_name] = dict()
            for i in range(len(model_param_search_item_combination)):
                search_key = keys[i]
                search_item = model_param_search_item_combination[i]
                curr_conf[model_specific_gs_params_name][search_key] = search_item

            # conf established, launch training
            conf = Config(curr_conf)
            all_search_confs.append(conf)
            curr_conf_idx = len(all_search_confs)
            now = time.time()
            logging.info(
                f"{model_type.name} grid search {curr_conf_idx}/{total_search_items} starts. curr time usage: {now - all_start_time:.2f}s")
            logging.info(
                f'{model_type.name} grid search conf {curr_conf_idx}/{total_search_items} = {json.dumps(conf.to_dict(), indent="    ")}'
            )

            model = init_model()
            model = model.to(conf.device)
            
            single_train_loop(model_type, conf, preprocessed_train_ds, preprocessed_eval_ds, model, compute_eval_metrics, to_device)
            train_over_time = time.time()
            logging.info(
                f"{model_type.name} grid search {curr_conf_idx}/{total_search_items} done. training duration: {train_over_time - now:.2f}s")



def calculate_metrics(start, 
                      model_type:ModelType, 
                      curr_train_step:int, 
                      model:MModule,
                      conf: Config,
                      compute_eval_metrics: Callable[[List, List, float], Dict],
                      preprocessed_eval_ds: MDataset,
                      to_device: Callable[[Any, Any], Tuple[Any, Any]],
                        loss_value: float,
                        train_records: Dict,
                      ):
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

def single_train_loop(model_type: ModelType,
                      conf: Config,
                      preprocessed_train_ds: MDataset,
                      preprocessed_eval_ds: MDataset,
                      model: MModule,
                      compute_eval_metrics: Callable[[List, List, float], Dict],
                      to_device: Callable[[Any, Any], Tuple[Any, Any]],
                      ):
    model_ckpts_dir = ckpts_dir / conf.dataset_environment_str/ model_type.name
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
                calculate_metrics(start, model_type, curr_train_step, model, conf, compute_eval_metrics, preprocessed_eval_ds, to_device, loss_value, train_records)
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
    logging.info(f"Saving model at step {curr_steps} with loss {curr_loss_value},save path: {save_path}")
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


def save_train_plot(conf, train_records, save_path, loss_step=200):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    # plot loss
    ax = axes[0]
    # todo, 采样
    losses = []
    for i in range(0, len(train_records['loss']), loss_step):
        losses.append(train_records['loss'][i])
    ax.plot([ i for i in range(len(losses))],losses, label="loss")
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