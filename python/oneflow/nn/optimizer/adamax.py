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
from typing import Callable, Dict, Iterator, List, Union, Tuple

import math

import oneflow as flow
from oneflow.nn.parameter import Parameter

from .optimizer import Optimizer, ParamGroup


class Adamax(Optimizer):
    r"""Implements Adamax algorithm. (a variant of Adam based on infinity norm).

    the equation of parameters updating is:

    .. math::

        & m_t = \beta_1*m_{t-1} + (1-\beta_1)*grad

        & u_t = \max(\beta_2u_{t-1}, |grad| + \epsilon)

        & param_{new} = param_{old} - learning\_rate * \frac{m_t}{(1 - \beta_1^t) u_t}


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing (default: False)
        fused (bool, optional): whether to divide all the parameters into several groups, then
            update each group of parameters with the fused kernel. (default: False)

    For example:

    Example 1:

    .. code-block:: python

        # Assume net is a custom model.
        adamax = flow.optim.Adamax(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on.
            # ...
            loss.backward()
            adamax.step()
            adamax.zero_grad()

    Example 2:

    .. code-block:: python

        # Assume net is a custom model.
        adamax = flow.optim.Adamax(
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
            adamax.clip_grad()
            adamax.step()
            adamax.zero_grad()

    If you want to use clip_grad, you can refer this example.

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`.

    """
    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.002,
        betas: Tuple[float] = (0.9, 0.999),
        eps=1e-8,
        weight_decay: float = 0.0,
        *,
        do_bias_correction: bool = True,
        maximize: bool = False,
        fused: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert weight_decay >= 0.0, f"Invalid weight_decay: {weight_decay}"
        assert fused is False, f"fused Adamax not implemented!"
        options = dict()
        options["lr"] = lr
        options["betas"] = betas
        options["eps"] = eps
        options["weight_decay"] = weight_decay
        options["do_bias_correction"] = do_bias_correction
        options["maximize"] = maximize
        options["fused"] = fused
        super().__init__(params, options)

        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()

                if param_group["fused"] and not param.is_cuda:
                    warnings.warn("Fused Adamax only support cuda parameters.")
                    param_group["fused"] = False

        self._adamax = (
            flow.stateful_op("adamax_update")
            .Input("model")
            .Input("model_diff")
            .Input("m")
            .Input("norm")
            .Build()
        )

    def _single_tensor_update(self, param_group):
        for param in param_group.parameters:
            if param.grad is None:
                continue
            if "exp_avg" not in self._state[param]:
                self._state[param]["exp_avg"] = flow.zeros_like(param)
            if "exp_inf" not in self._state[param]:
                self._state[param]["exp_inf"] = flow.zeros_like(param)
            m_tensor = self._state[param]["exp_avg"]
            norm_tensor = self._state[param]["exp_inf"]

            flow._C.dispatch_adamax_update(
                self._adamax,
                (param, param.grad, m_tensor, norm_tensor),
                learning_rate=param_group["lr"],
                l2=param_group["weight_decay"],
                beta1=param_group["betas"][0],
                beta2=param_group["betas"][1],
                epsilon=param_group["eps"],
                do_bias_correction=param_group["do_bias_correction"],
                bias_correction1=param_group["bias_correction1"],
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
                if param_group["do_bias_correction"]:
                    param_group["bias_correction1"] = 1.0 - math.pow(
                        param_group["betas"][0], self._state["step"] + 1
                    )
                else:
                    param_group["bias_correction1"] = 1.0
                self._single_tensor_update(param_group)

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
            l2 = param_group["weight_decay"]
            beta1 = param_group["betas"][0]
            beta2 = param_group["betas"][1]
            epsilon = param_group["eps"]
            do_bias_correction = param_group["do_bias_correction"]
            maximize = param_group["maximize"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.adamax_conf.beta1 = beta1
            optimizer_conf.adamax_conf.beta2 = beta2
            optimizer_conf.adamax_conf.epsilon = epsilon
            optimizer_conf.adamax_conf.do_bias_correction = do_bias_correction
            optimizer_conf.adamax_conf.maximize = maximize

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs

    @property
    def support_sparse(self):
        """Whether Adamax Optimizer support sparse update.

        """
        return True
