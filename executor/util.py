import numpy as np
from typing import Mapping

def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    if isinstance(tensors, np.ndarray):
        return tensors
    return tensors.detach()