from objects import ModelType
from .grouping_based_executor import MLPTest_GroupingBasedExecutor
from .op_based_executor import MLP_OPBasedExecutor, PerfNet_OPBasedExecutor, GBDT_OPBasedExecutor
from .subgraph_based_executor import MLPTest_SubgraphBasedExecutor, TransformerRegressionSubgraphBasedExecutor


def init_executor(conf):
    return {
        ModelType.GBDT: GBDT_OPBasedExecutor,
        ModelType.MLP: MLP_OPBasedExecutor,
        ModelType.PerfNet: PerfNet_OPBasedExecutor,
        ModelType.MLPTestGrouping: MLPTest_GroupingBasedExecutor,
        ModelType.MLPTestSubgraph: MLPTest_SubgraphBasedExecutor,
        ModelType.TransformerRegression: TransformerRegressionSubgraphBasedExecutor,
    }[conf.model_type](conf)
