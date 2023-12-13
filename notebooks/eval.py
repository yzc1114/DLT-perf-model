import logging
from typing import Dict, List

from ckpt_loader import load_train_records, TrainRecord
from logger import init_logging
from objects import ModelType, GPUType


init_logging()

# gpus = [gpu_type for gpu_type in GPUType]
gpus = [GPUType.T4_CPUALL]
train_models = [model_type for model_type in ModelType]





def get_metirc_best_records(gpu_records:Dict[ModelType, List[TrainRecord]],metric_name="MSE") -> Dict[ModelType, TrainRecord]:
    metric_best_records = dict()
    for train_model in train_models:
        metric_best_records[train_model] = None
        for record in gpu_records[train_model]:
            record_eval_metric = record.optimal_eval_metric(metric_name).metrics[metric_name]
            if metric_best_records[train_model] is None:
                metric_best_records[train_model] = record
                continue
            best_record_eval_metric = metric_best_records[train_model].optimal_eval_metric(metric_name).metrics[metric_name]
            if record_eval_metric < best_record_eval_metric:
                metric_best_records[train_model] = record
    return metric_best_records


if __name__ == '__main__':
    records: Dict[GPUType, Dict[ModelType, List[TrainRecord]]] = dict()
    best_records : Dict[GPUType, TrainRecord] = dict()
    metric_name = 'MRE'
    for gpu in gpus:
        logging.info(f"GPU: {gpu.name}")
        records[gpu] = load_train_records(gpu, train_models)
        best_records[gpu] = get_metirc_best_records(records[gpu], metric_name = metric_name)
        logging.info(f"Best records for metric {metric_name}")
        for model_type, record in best_records[gpu].items():
            if record is None:
                continue
            logging.info(f"Model: {model_type.name}, metric: {metric_name}, value: {record.optimal_eval_metric(metric_name).metrics[metric_name]}")
