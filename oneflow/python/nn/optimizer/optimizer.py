"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, Callable, Union, Any, Iterator
from types import GeneratorType

from oneflow.python.nn.parameter import Parameter
from oneflow.python.framework.tensor import Tensor


class ParamGroup(object):
    def __init__(
        self,
        parameters: Union[Iterator[Parameter], Dict[str, Any]],
        default_options: Dict,
    ):
        if isinstance(parameters, GeneratorType):
            self._parameters = list(parameters)
            self._options = default_options
        else:  # Dict
            assert "param" in parameters
            self._parameters = list(parameters["param"])
            self._options = default_options
            for key in self._options:
                if key in parameters:
                    self._options[key] = parameters[key]

    @property
    def options(self):
        return self._options

    @property
    def parameters(self):
        return self._parameters


class Optimizer(object):
    def __init__(self):
        self._param_groups = list()
        self._default_options = dict()
        self._state = dict()
        self._state["step"] = 0
        self._op = None

    def add_param_group(self, param_group) -> None:
        # TODO(wyg)
        raise NotImplementedError()

    def load_state_dict(self, state_dict) -> None:
        # TODO(wyg)
        raise NotImplementedError()

    def state_dict(self):
        # TODO(wyg)
        raise NotImplementedError()

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        raise NotImplementedError()

    def zero_grad(self, set_to_none: bool = False):
        for param_group in self._param_groups:
            for param in param_group.parameters:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.fill_(0)
                    # param.grad.zeros_()
