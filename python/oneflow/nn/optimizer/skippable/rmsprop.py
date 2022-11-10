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

import oneflow as flow
from oneflow.nn.parameter import Parameter

from oneflow.nn.optimizer.optimizer import ParamGroup
from oneflow.nn.optimizer.rmsprop import RMSprop as NNRMSprop


class RMSprop(NNRMSprop):
    """Implements RMSprop algorithm.

    oot Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        W = w - \\frac{\\eta} {\\\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.
    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    if centered is True:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        g(w, t) = \\alpha g(w, t-1) + (1 - \\alpha)\\nabla Q_{i}(w)

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) - (g(w, t))^2 +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    where, :math:`\\alpha` is a hyperparameter and typical values are 0.99, 0.95
    and so on. :math:`\\beta` is the momentum term. :math:`\\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0, oneflow not support momenmtum > 0 now!)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    For example: 

    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        rmsprop = flow.optim.RMSprop(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

    Example 2: 

    .. code-block:: python 

        # Assume net is a custom model. 
        rmsprop = flow.optim.RMSprop(
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
            rmsprop.clip_grad()
            rmsprop.step()
            rmsprop.zero_grad()

    If you want to use clip_grad, you can refer this example. 

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 

    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered)

        self._state["step"] = flow.tensor(0)
        self._centered_rmsprop = (
            flow.stateful_op("rmsprop_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Input("mean_square")
            .Input("mean_gradient")
            .Build()
        )
        self._rmsprop = (
            flow.stateful_op("rmsprop_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Input("mean_square")
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
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "epsilon": param_group["eps"],
                    "decay_rate": param_group["alpha"],
                    "l2": param_group["weight_decay"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue

                    if "square_avg" not in self._state[param]:
                        self._state[param]["square_avg"] = flow.zeros_like(param)
                    ms_tensor = self._state[param]["square_avg"]

                    if param_group["centered"]:
                        if "grad_avg" not in self._state[param]:
                            self._state[param]["grad_avg"] = flow.zeros_like(param)
                        mg_tensor = self._state[param]["grad_avg"]
                        flow._C.dispatch_rmsprop_update(
                            self._centered_rmsprop,
                            (
                                param,
                                param.grad,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                ms_tensor,
                                mg_tensor,
                            ),
                            centered=True,
                            **kwargs,
                        )
                    else:
                        flow._C.dispatch_rmsprop_update(
                            self._rmsprop,
                            (
                                param,
                                param.grad,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                ms_tensor,
                            ),
                            **kwargs,
                        )
            self._state["step"] += skip_condition
            return loss
