from abc import ABC
from abc import abstractmethod

import torch.nn


class MModule(torch.nn.Module, ABC):
    @abstractmethod
    def loss(self, inputs):
        pass
