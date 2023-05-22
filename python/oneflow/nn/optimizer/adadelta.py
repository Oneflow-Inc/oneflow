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
from oneflow.optim.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter


class Adadelta(Optimizer):
    r"""Implements Adadelta Optimizer. 

        The formula is: 

        .. math::

            & v_{t} = v_{t-1} * rho + g_{t}^2 * (1 - rho)

            & delta = \frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_{t} + \epsilon}} * g_{t}
            
            & u_{t} = u_{t-1} * rho + delta^2*(1 - rho)

            & x_{t} = x_{t-1} - lr * delta

        Args:
            params (Union[Iterator[Parameter], List[Dict]]): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): The learning rate. Defaults to 0.001.
            rho (float, optional): The decay factor of learning rate. Defaults to 0.0.
            eps (float, optional): A small constant terms added to the denominator to improve numerical stability. Defaults to 1e-10.
            weight_decay (float, optional): The weight decay. Defaults to 0.
            maximize (bool, optional): maximize the params based on the objective, instead of minimizing. Defaults False.
            contiguous_params (bool, optional): whether to use contiguous ParamGroup 
                which puts all parameters of the same type, device and group into the
                same tensor and update them together. (default: False)
        
        For example: 

        Example 1: 

        .. code-block:: python

            # Assume net is a custom model. 
            adadelta = flow.optim.Adadelta(net.parameters(), lr=1e-3)

            for epoch in range(epochs):
                # Read data, Compute the loss and so on. 
                # ...
                loss.backward()
                adadelta.step()
                adadelta.zero_grad()

        Example 2: 

        .. code-block:: python 

            # Assume net is a custom model. 
            adadelta = flow.optim.Adadelta(
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
                adadelta.clip_grad()
                adadelta.step()
                adadelta.zero_grad()

        If you want to use clip_grad, you can refer this example. 

        For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 
        
    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        maximize: bool = False,
        contiguous_params: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert 1.0 >= rho >= 0.0, f"Invalid rho value: {rho}"
        assert (
            not maximize
        ), f"In Graph Mode, weight decay has been added to Variable, it cause different result with Eager Mode when maximize = True"
        options = dict()
        options["lr"] = lr
        options["rho"] = rho
        options["eps"] = eps
        options["maximize"] = maximize
        options["weight_decay"] = weight_decay
        options["contiguous_params"] = contiguous_params
        super().__init__(params, options)

        for param_group in self.param_groups:
            if param_group["contiguous_params"]:
                param_list = param_group.contiguous_parameters
            else:
                param_list = param_group.parameters

            for param in param_list:
                assert param.is_leaf, "parameters must be leaf tensor"
                self.state[param] = dict()
                self.state[param]["square_avgs"] = flow.zeros_like(param)
                self.state[param]["acc_deltas"] = flow.zeros_like(param)

        self._op = (
            flow.stateful_op("adadelta_update")
            .Input("model")
            .Input("model_diff")
            .Input("square_avgs")
            .Input("acc_deltas")
            .Build()
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
                with flow.enable_grad():
                    loss = closure()

            for param_group in self.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "l2": param_group["weight_decay"],
                    "rho": param_group["rho"],
                    "epsilon": param_group["eps"],
                    "maximize": param_group["maximize"],
                }

                if param_group["contiguous_params"]:
                    param_list = param_group.contiguous_parameters
                else:
                    param_list = param_group.parameters

                for param in param_list:
                    if param.grad is None:
                        continue
                    square_avgs_tensor = self.state[param]["square_avgs"]
                    acc_deltas_tensor = self.state[param]["acc_deltas"]
                    flow._C.dispatch_adadelta_update(
                        self._op,
                        (param, param.grad, square_avgs_tensor, acc_deltas_tensor),
                        **kwargs,
                    )

            self.state["step"] = self.state["step"] + 1
            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            assert (
                param_group["contiguous_params"] != True
            ), "contiguous_params cannot be used in graph"

            optimizer_conf = train_conf.optimizer_conf.add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            l2 = param_group["weight_decay"]
            rho = param_group["rho"]
            epsilon = param_group["eps"]
            maximize = param_group["maximize"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.adadelta_conf.rho = rho
            optimizer_conf.adadelta_conf.epsilon = epsilon
            optimizer_conf.adadelta_conf.maximize = maximize

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs
