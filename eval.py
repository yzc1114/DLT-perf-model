from typing import Dict, List

from ckpt_loader import load_train_records, TrainRecord
from logger import init_logging
from objects import ModelType

init_logging()

train_models = [
    ModelType.MLP,
    ModelType.PerfNet,
    ModelType.Transformer,
    ModelType.LSTM,
    ModelType.GRU,
    ModelType.GCNSubgraph,
    ModelType.MLPTestSubgraph,
    ModelType.GCNGrouping,
    ModelType.MLPTestGrouping
]

records: Dict[ModelType, List[TrainRecord]] = load_train_records(train_models)


def get_best_records(metric_name="MSE") -> Dict[ModelType, TrainRecord]:
    best_records = dict()
    for train_model in train_models:
        best_records[train_model] = None
        for record in records[train_model]:
            record_eval_metric = record.optimal_eval_metric(metric_name).metrics[metric_name]
            if best_records[train_model] is None:
                best_records[train_model] = record_eval_metric
                continue
            best_record_eval_metric = best_records[train_model].optimal_eval_metric(metric_name).metrics[metric_name]
            if record_eval_metric < best_record_eval_metric:
                best_records[train_model] = record_eval_metric
    print(best_records)
    return best_records


if __name__ == '__main__':
    get_best_records('MRE')
    # get_best_records('MAE')
    get_best_records('RMSE')
    # get_best_records('train_loss')
    # get_best_records('eval_loss')