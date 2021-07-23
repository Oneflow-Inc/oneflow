from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        self._data = data
        self._data.requires_grad = requires_grad

    def __getattr__(self, name):
        return getattr(self._data, name)
