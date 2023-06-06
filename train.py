import logging
import time
from itertools import product

import numpy as np
import json

from objects import ModelType

from config import Config, configs
from executor import get_executor_cls
from logger import init_logging

init_logging()


def launch_grid_search(train_model: ModelType):
    conf_dict = configs[train_model]
    executor_cls = get_executor_cls(model_type=train_model)

    grid_search_params = dict()
    for k, v in conf_dict.items():
        if not isinstance(v, list):
            grid_search_params[k] = [v]
        else:
            grid_search_params[k] = v
    grid_search_item_lens = list((len(v) for v in grid_search_params.values()))
    for k, v in executor_cls.grid_search_model_params().items():
        grid_search_item_lens.append(len(v))
    total_search_items = np.prod(grid_search_item_lens)

    logging.info(f"{train_model} grid search total search items: {total_search_items}")

    search_param_keys = sorted(grid_search_params.keys())
    search_items = [grid_search_params[k] for k in search_param_keys]

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
        grid_search_model_params = executor_cls.grid_search_model_params()
        model_param_search_keys = sorted(grid_search_model_params.keys())
        model_param_search_items = [grid_search_model_params[k] for k in model_param_search_keys]
        for model_param_search_item_combination in list(product(*model_param_search_items)):
            curr_conf["model_params"] = dict()
            for i in range(len(model_param_search_item_combination)):
                search_model_param_key = model_param_search_keys[i]
                search_model_item = model_param_search_item_combination[i]
                curr_conf["model_params"][search_model_param_key] = search_model_item

            # conf established, launch training
            conf = Config(curr_conf)
            all_search_confs.append(conf)
            curr_conf_idx = len(all_search_confs)
            now = time.time()
            logging.info(
                f"{train_model} grid search {curr_conf_idx}/{total_search_items} starts. curr time usage: {now - all_start_time:.2f}s")
            logging.info(
                f'{train_model} grid search conf {curr_conf_idx}/{total_search_items} = {json.dumps(conf.to_dict(), indent="    ")}'
            )
            executor = executor_cls(conf=Config(curr_conf))
            # executor.train()
            train_over_time = time.time()
            logging.info(
                f"{train_model} grid search {curr_conf_idx}/{total_search_items} done. training duration: {train_over_time - now:.2f}s")


def launch_single_train(train_model: ModelType):
    conf_dict = configs[train_model]
    confirmed_params = dict()
    for k, v in conf_dict.items():
        if not isinstance(v, list):
            confirmed_params[k] = v
        else:
            confirmed_params[k] = v[0]
    now = time.time()
    logging.info(f"{train_model} single train starts. conf = {json.dumps(confirmed_params, indent='    ')}")
    conf = Config.from_dict(confirmed_params)
    executor_cls = get_executor_cls(model_type=train_model)
    executor = executor_cls(conf)
    executor.train()
    train_over_time = time.time()
    logging.info(f"{train_model} single train ends. training duration = {train_over_time - now:.2f}s.")


def launch_train(mode="single-train"):
    if mode == "single-train":
        launch_func = launch_single_train
    elif mode == "grid-search":
        launch_func = launch_grid_search
    else:
        raise ValueError(f"Unknown launch mode: {mode}")
    for i, train_model in enumerate(train_models):
        now = time.time()
        logging.info(f"launching {train_model} with {mode} starts. rest models: {[m.name for m in train_models[i+1:]]}")
        launch_func(train_model)
        launch_over = time.time()
        logging.info(f"launching {train_model} with {mode} ends. launching duration = {launch_over - now:.2f}s.")


train_models = [
    ModelType.MLP,
    ModelType.GBDT,
    ModelType.GCNSubgraph,
    ModelType.PerfNet,
    ModelType.Transformer,
    ModelType.LSTM,
    ModelType.GCNGrouping
]



if __name__ == '__main__':
    launch_train(mode="single-train")
    # launch_train(mode="grid-search")
