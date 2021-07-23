import oneflow as flow
from oneflow.framework.tensor import Tensor

class Parameter(Tensor):

    def __init__(self, data, requires_grad=True):
        if not isinstance(data, Tensor):
            data = Tensor(data)
        self._data = data
        self._data.requires_grad = requires_grad

    def __getattr__(self, name):
        return getattr(self._data, name)