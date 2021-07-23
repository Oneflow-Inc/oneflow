import collections
import warnings
from typing import Any, Callable, Dict, Iterator, Union

from oneflow.compatible.single_client.python.framework.tensor import Tensor
from oneflow.compatible.single_client.python.nn.parameter import Parameter


class ParamGroup(object):
    def __init__(
        self,
        parameters: Union[Iterator[Parameter], Dict[str, Any]],
        default_options: Dict,
    ):
        if isinstance(parameters, collections.abc.Iterator):
            self._parameters = list(parameters)
            self._options = default_options
        else:
            assert "params" in parameters
            self._parameters = list(parameters["params"])
            self._options = default_options
            for key in self._options:
                if key in parameters:
                    self._options[key] = parameters[key]

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters
