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

from .optimizer import Optimizer, ParamGroup


class SGD(Optimizer):
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
        weight_decay: float = 0.0,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert momentum >= 0.0, f"Invalid momentum: {momentum}"
        assert weight_decay >= 0.0, f"Invalid weight_decay: {weight_decay}"
        options = dict()
        options["lr"] = lr
        options["momentum"] = momentum
        options["weight_decay"] = weight_decay
        super().__init__(params, options)

        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()

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
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with flow.no_grad():
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
                        flow._C.dispatch_sgd_update(
                            self._sgd, (param, param.grad), learning_rate=lr, l2=l2
                        )
                    else:
                        if "momentum_buf" not in self._state[param]:
                            self._state[param]["momentum_buf"] = flow.zeros_like(param)
                        momentum_buf = self._state[param]["momentum_buf"]
                        beta = param_group["momentum"]
                        flow._C.dispatch_momentum_update(
                            self._momentum_sgd,
                            (param, param.grad, momentum_buf),
                            learning_rate=lr,
                            l2=l2,
                            beta=beta,
                        )
            self._state["step"] = self._state["step"] + 1
            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            optimizer_conf = train_conf.optimizer_conf.add()
            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            beta = param_group["momentum"]
            l2 = param_group["weight_decay"]

            optimizer_conf.base_learning_rate = lr
            if beta == 0:
                optimizer_conf.naive_conf.SetInParent()
            else:
                optimizer_conf.momentum_conf.beta = beta

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs

    @property
    def support_sparse(self):
        """Whether SGD Optimizer support sparse update. 

        """
        return True
