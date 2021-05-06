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

from typing import List, Dict, Callable, Union, Any, Iterator
from types import GeneratorType

import numpy as np
import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export
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


@oneflow_export("optim.SGD")
class SGD(Optimizer):
    r"""
    TODO
    """

    def __init__(
        self,
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float,
        momentum: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert momentum >= 0.0, f"Invalid momentum: {momentum}"
        assert scale >= 0.0, f"Invalid scale factor: {scale}"

        self._default_options = dict()
        self._default_options["lr"] = lr
        self._default_options["scale"] = scale
        if momentum != 0.0:
            self._default_options["momentum"] = momentum

        # Add parameters
        if isinstance(parameters, GeneratorType):
            self._param_groups.append(ParamGroup(parameters, self._default_options))
        else:  # List[Dict]
            for param in parameters:
                self._param_groups.append(ParamGroup(param, self._default_options))

        for param_group in self._param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
                if "momentum" in self._default_options:
                    self._state[param]["momentum_buf"] = flow.tmp.zeros(
                        # TODO: zeros module support flow.Size parameter
                        tuple(param.shape)
                    )

        if "momentum" in self._default_options.keys():
            self._op = (
                flow.builtin_op("momentum_update")
                .Input("model")
                .Input("model_diff")
                .Input("learning_rate")
                .Input("momentum")
                .Attr("scale", self._default_options["scale"])
                .Attr("l1", 0.0)
                .Attr("l2", 0.0)
                .Attr("beta", self._default_options["momentum"])
                .Attr("weight_decay", 0.0)
                .Build()
            )
        else:
            self._op = (
                flow.builtin_op("sgd_update")
                .Input("model")
                .Input("model_diff")
                .Input("learning_rate")
                .Attr("scale", self._default_options["scale"])
                .Attr("weight_decay", 0.0)
                .Attr("l1", 0.0)
                .Attr("l2", 0.0)
                .Build()
            )

    def step(self, closure: Callable = None):
        with flow.no_grad():
            loss = None
            if closure is not None:
                loss = closure()

            for param_group in self._param_groups:
                lr_tensor = flow.Tensor([param_group.options["lr"]])
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "momentum" in self._default_options:
                        momentum_buf = self._state[param]["momentum_buf"]
                        self._op(param, param.grad, lr_tensor, momentum_buf)
                    else:
                        self._op(param, param.grad, lr_tensor)

            self._state["step"] = self._state["step"] + 1
            return loss
