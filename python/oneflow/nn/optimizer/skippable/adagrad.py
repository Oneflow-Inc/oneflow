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
import math
from typing import Callable, Dict, Iterator, List, Tuple, Union

import oneflow as flow
from oneflow.nn.parameter import Parameter

from oneflow.nn.optimizer.optimizer import ParamGroup
from oneflow.nn.optimizer.adagrad import Adagrad as NNAdagrad


class Adagrad(NNAdagrad):
    r"""Implements Adagrad Optimizer. 

        The formula is: 

        .. math:: 

            & S_{t} = S_{t-1} + grad \odot grad 
            
            & decay\_lr = \frac{learning\_rate}{(1 + (train\_step - 1) * lr\_decay)}

            & X_{t} = X_{t-1} - \frac{decay\_lr}{\sqrt{S_{t} + \epsilon}} \odot grad

        Args:
            params (Union[Iterator[Parameter], List[Dict]]): iterable of parameters to optimize or dicts defining
            parameter groups
            lr (float, optional): The learning rate. Defaults to 0.001.
            lr_decay (float, optional): The decay factor of learning rate. Defaults to 0.0.
            weight_decay (float, optional): The weight decay. Defaults to 0.
            initial_accumulator_value (float, optional): The initial value of S. Defaults to 0.0.
            eps (float, optional): A small constant terms added to the denominator to improve numerical stability. Defaults to 1e-10.
        
        For example: 

        Example 1: 

        .. code-block:: python

            # Assume net is a custom model. 
            adagrad = flow.optim.Adagrad(net.parameters(), lr=1e-3)

            for epoch in range(epochs):
                # Read data, Compute the loss and so on. 
                # ...
                loss.backward()
                adagrad.step()
                adagrad.zero_grad()

        Example 2: 

        .. code-block:: python 

            # Assume net is a custom model. 
            adagrad = flow.optim.Adagrad(
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
                adagrad.clip_grad()
                adagrad.step()
                adagrad.zero_grad()

        If you want to use clip_grad, you can refer this example. 

        For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 
        
        """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        lr_decay: float = 0.0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ):
        super().__init__(
            params, lr, lr_decay, weight_decay, initial_accumulator_value, eps
        )

        self._state["step"] = flow.tensor(0)
        self._op = (
            flow.stateful_op("adagrad_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Input("train_step")
            .Input("sum")
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
                    "l2": param_group["weight_decay"],
                    "epsilon": param_group["eps"],
                    "lr_decay": param_group["lr_decay"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    sum_tensor = self._state[param]["sum"]
                    flow._C.dispatch_adagrad_update(
                        self._op,
                        (
                            param,
                            param.grad,
                            skip_condition.to(param.device)
                            if param.is_local
                            else skip_condition.to_global(
                                param.placement,
                                [flow.sbp.broadcast for _ in range(len(param.sbp))],
                            ),
                            (
                                (self._state["step"] + 1).to(param.device)
                                if param.is_local
                                else (self._state["step"] + 1).to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                    check_meta=False,
                                )
                            ),
                            sum_tensor,
                        ),
                        **kwargs,
                    )

            self._state["step"] += skip_condition
            return loss
