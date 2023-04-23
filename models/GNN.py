from graph import Graph
from .base import PerfPredictor

class GCNPredictor(PerfPredictor):
    def predict(self, graph: Graph) -> float:
        pass