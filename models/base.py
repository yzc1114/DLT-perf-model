from abc import ABC, abstractmethod
from graph import Graph


class PerfPredictor(ABC):
    @abstractmethod
    def predict(self, graph: Graph) -> float:
        pass

