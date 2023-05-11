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


class Adagrad(Optimizer):
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
            contiguous_params (bool, optional): whether to use contiguous ParamGroup 
                which puts all parameters of the same type, device and group into the
                same tensor and update them together. (default: False)
        
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
        contiguous_params: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert (
            initial_accumulator_value >= 0.0
        ), f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"

        options = dict()
        options["lr"] = lr
        options["initial_accumulator_value"] = initial_accumulator_value
        options["lr_decay"] = lr_decay
        options["weight_decay"] = weight_decay
        options["eps"] = eps
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
                self.state[param]["sum"] = flow.zeros_like(param).fill_(
                    param_group["initial_accumulator_value"]
                )

        self._op = (
            flow.stateful_op("adagrad_update")
            .Input("model")
            .Input("model_diff")
            .Input("sum")
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
                    "epsilon": param_group["eps"],
                    "lr_decay": param_group["lr_decay"],
                    "train_step_val": self.state["step"] + 1,
                }

                if param_group["contiguous_params"]:
                    param_list = param_group.contiguous_parameters
                else:
                    param_list = param_group.parameters

                for param in param_list:
                    if param.grad is None:
                        continue
                    sum_tensor = self.state[param]["sum"]
                    flow._C.dispatch_adagrad_update(
                        self._op, (param, param.grad, sum_tensor), **kwargs
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
            initial_accumulator_value = param_group["initial_accumulator_value"]
            lr_decay = param_group["lr_decay"]
            epsilon = param_group["eps"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.adagrad_conf.initial_accumulator_value = (
                initial_accumulator_value
            )
            optimizer_conf.adagrad_conf.lr_decay = lr_decay
            optimizer_conf.adagrad_conf.epsilon = epsilon

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs
