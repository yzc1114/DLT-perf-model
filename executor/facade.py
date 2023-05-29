
from objects import ModelType
from .op_based_executor import MLP_OPBasedExecutor, PerfNet_OPBasedExecutor, GBDT_OPBasedExecutor
from .grouping_based_executor import MLPTest_GroupingBasedExecutor


def init_executor(conf):
    return {
        ModelType.GBDT: GBDT_OPBasedExecutor,
        ModelType.MLP: MLP_OPBasedExecutor,
        ModelType.PerfNet: PerfNet_OPBasedExecutor,
        ModelType.MLPTestGrouping: MLPTest_GroupingBasedExecutor
    }[conf.model_type](conf)