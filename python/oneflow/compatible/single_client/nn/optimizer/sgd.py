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
import collections
from typing import Callable, Dict, Iterator, List, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.parameter import Parameter

from .optimizer import Optimizer, ParamGroup


class SGD(Optimizer):
    """Implements SGD algorithm.

    This algorithm takes a random sample’s gradient as an approximate estimate of the overall gradient in small batch gradient descent.

    When the momentum = 0, the equation of parameters updating is:

        .. math::

            param_{new} = param_{old} - learning\\_rate * grad

    With momentum, the equation of parameters updating is:

        .. math::

            & V_t = \\beta * V_{t-1} + learning\\_rate * g_t

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
        lr: float = 0.001,
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
        if isinstance(parameters, collections.abc.Iterator):
            self.param_groups.append(ParamGroup(parameters, self._default_options))
        else:
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
            flow.stateful_op("momentum_update")
            .Input("model")
            .Input("model_diff")
            .Input("momentum")
            .Build()
        )
        self._sgd = (
            flow.stateful_op("sgd_update").Input("model").Input("model_diff").Build()
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
                        flow._C.dispatch_sgd_update(
                            self._sgd,
                            (param, param.grad),
                            learning_rate=lr,
                            scale=scale,
                        )
                    else:
                        momentum_buf = self._state[param]["momentum_buf"]
                        scale = param_group["scale"]
                        beta = param_group["momentum"]
                        flow._C.dispatch_momentum_update(
                            self._momentum_sgd,
                            (param, param.grad, momentum_buf),
                            learning_rate=lr,
                            scale=scale,
                            beta=beta,
                        )
            self._state["step"] = self._state["step"] + 1
            return loss
