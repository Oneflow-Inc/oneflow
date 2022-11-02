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
import warnings
from typing import Callable, Dict, Iterator, List, Union

import oneflow as flow
from oneflow.nn.parameter import Parameter

from oneflow.nn.optimizer.sgd import SGD as NNSGD


class SGD(NNSGD):
    """Implements SGD algorithm.

    This algorithm takes a random sample’s gradient as an approximate estimate of
    the overall gradient in small batch gradient descent.

    When the momentum = 0, the equation of parameters updating is:

        .. math::

            param_{new} = param_{old} - learning\\_rate * grad

    With momentum, the equation of parameters updating is:

        .. math::

            & V_t = \\beta * V_{t-1} - learning\\_rate * (g_t + param_{old} * weight\\_decay)

            & param_{new} = param_{old} + V_t

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): Momentum factor (default: 0.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)

    For example: 

    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        sgd = flow.optim.SGD(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            sgd.step()
            sgd.zero_grad()

    Example 2: 

    .. code-block:: python 

        # Assume net is a custom model. 
        sgd = flow.optim.SGD(
            [
                {
                    "params": net.parameters(),
                    "lr": learning_rate,
                    "clip_grad_max_norm": 0.5,
                    "clip_grad_norm_type": 2.0,
                }
            ],
        )

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            sgd.clip_grad()
            sgd.step()
            sgd.zero_grad()

    If you want to use clip_grad, you can refer this example. 

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 

    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        super().__init__(
            params, lr, momentum, dampening, weight_decay, nesterov, maximize
        )

        self._state["step"] = flow.tensor(0)
        self._momentum_sgd = (
            flow.stateful_op("momentum_update")
            .Input("model")
            .Input("model_diff")
            .Input("momentum")
            .Input("skip_if")
            .Build()
        )
        self._sgd = (
            flow.stateful_op("sgd_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Build()
        )

    def step(self, skip_condition=False, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert isinstance(
            skip_condition, (flow.Tensor, bool)
        ), "skip_condition must be  a scalar bool Tensor ora boolean value!"
        if isinstance(skip_condition, flow.Tensor):
            assert (
                skip_condition.dtype is flow.bool and skip_condition.dim() == 0
            ), "skip_condition must be  a scalar bool Tensor ora boolean value!"
        with flow.no_grad():
            if isinstance(skip_condition, bool):
                skip_condition = flow.tensor(skip_condition, dtype=flow.int64)
            else:
                skip_condition = skip_condition.to(flow.int64)
            loss = None
            if closure is not None:
                loss = closure()
            for param_group in self.param_groups:
                lr = param_group["lr"]
                l2 = param_group["weight_decay"]
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if param_group["momentum"] == 0.0:
                        # TODO: Support param `maximize` in Naive SGD Optimizer. (zhengzekang)
                        flow._C.dispatch_sgd_update(
                            self._sgd,
                            (
                                param,
                                param.grad,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                            ),
                            learning_rate=lr,
                            l2=l2,
                        )
                    else:
                        if "momentum_buf" not in self._state[param]:
                            self._state[param]["momentum_buf"] = flow.zeros_like(param)
                        momentum_buf = self._state[param]["momentum_buf"]
                        beta = param_group["momentum"]
                        dampening = param_group["dampening"]
                        nesterov = param_group["nesterov"]
                        maximize = param_group["maximize"]
                        flow._C.dispatch_momentum_update(
                            self._momentum_sgd,
                            (
                                param,
                                param.grad,
                                momentum_buf,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                            ),
                            learning_rate=lr,
                            l2=l2,
                            beta=beta,
                            dampening=dampening,
                            nesterov=nesterov,
                            maximize=maximize,
                        )
            self._state["step"] += skip_condition
            return loss
