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

import warnings
from typing import Dict, Callable, Union, Any, Iterator
from types import GeneratorType

from oneflow.python.oneflow_export import oneflow_export, experimental_api
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


@oneflow_export("optim.Optimizer")
@experimental_api
class Optimizer(object):
    def __init__(self):
        self.param_groups = list()
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
        all_grad_is_none = True
        for param_group in self.param_groups:
            for param in param_group.parameters:
                if param.grad is not None:
                    all_grad_is_none = False
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zeros_()
        if all_grad_is_none:
            # TODO: delete this after implementing Tensor.data
            warnings.warn(
                "\nParameters in optimizer do not have gradient.\n"
                "Please check `loss.backward()` is called or not,\n"
                "or try to declare optimizer after calling `module.to()`"
            )
