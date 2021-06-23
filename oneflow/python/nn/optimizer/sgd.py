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

from typing import List, Dict, Callable, Union, Iterator
from types import GeneratorType

import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.parameter import Parameter
from .optimizer import Optimizer, ParamGroup


@oneflow_export("optim.SGD")
@experimental_api
class SGD(Optimizer):
    r"""Implements SGD algorithm.

    This algorithm takes a random sample’s gradient as an approximate estimate of the overall gradient in small batch gradient descent.

    When the momentum = 0, the equation of parameters updating is:

        .. math::

            param_{new} = param_{old} - learning\_rate * grad

    With momentum, the equation of parameters updating is:

        .. math::

            & V_t = \beta * V_{t-1} + learning\_rate * g_t

            & param_{new} = param_{old} - V_t

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): Momentum factor (default: 0.0)
        scale (float, optional): the scale factor of loss (default: 1.0)

    """

    def __init__(
        self,
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float = 1e-3,
        momentum: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert momentum >= 0.0, f"Invalid momentum: {momentum}"
        assert scale >= 0.0, f"Invalid scale factor: {scale}"

        self._default_options["lr"] = lr
        self._default_options["scale"] = scale
        self._default_options["momentum"] = momentum

        # Add parameters
        if isinstance(parameters, GeneratorType):
            self.param_groups.append(ParamGroup(parameters, self._default_options))
        else:  # List[Dict]
            for param in parameters:
                self.param_groups.append(ParamGroup(param, self._default_options))

        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
                if param_group["momentum"] != 0.0:
                    self._state[param]["momentum_buf"] = flow.experimental.zeros_like(
                        param
                    )

        self._momentum_sgd = (
            flow.builtin_op("momentum_update")
            .Input("model")
            .Input("model_diff")
            .Input("momentum")
            .Attr("l1", 0.0)
            .Attr("l2", 0.0)
            .Attr("weight_decay", 0.0)
            .Build()
        )
        self._sgd = (
            flow.builtin_op("sgd_update")
            .Input("model")
            .Input("model_diff")
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

            for param_group in self.param_groups:
                lr = param_group["lr"]
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if param_group["momentum"] == 0.0:
                        scale = param_group["scale"]
                        self._sgd(param, param.grad, learning_rate_val=lr, scale=scale)
                    else:
                        momentum_buf = self._state[param]["momentum_buf"]
                        scale = param_group["scale"]
                        beta = param_group["momentum"]
                        self._momentum_sgd(
                            param,
                            param.grad,
                            momentum_buf,
                            learning_rate_val=lr,
                            scale=scale,
                            beta=beta,
                        )

            self._state["step"] = self._state["step"] + 1
            return loss
