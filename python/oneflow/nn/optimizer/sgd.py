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
from typing import Callable, Dict, Iterator, List, Union

import oneflow as flow
from oneflow.nn.parameter import Parameter

from ...optim.optimizer import Optimizer, ParamGroup


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
        contiguous_params (bool, optional): whether to use contiguous ParamGroup 
            which puts all parameters of the same type, device and group into the
            same tensor and update them together. (default: False)
        fused (bool, optional): whether to divide all the parameters into several groups, then
            update each group of parameters with the fused kernel. (default: False)

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
        contiguous_params: bool = False,
        fused: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert momentum >= 0.0, f"Invalid momentum: {momentum}"
        assert weight_decay >= 0.0, f"Invalid weight_decay: {weight_decay}"
        if maximize:
            warnings.warn(
                "Only Momentum > 0.0, param `maximize` takes effect. ", FutureWarning,
            )
        options = dict()
        options["lr"] = lr
        options["momentum"] = momentum
        options["dampening"] = dampening
        options["weight_decay"] = weight_decay
        options["nesterov"] = nesterov
        options["maximize"] = maximize
        options["contiguous_params"] = contiguous_params
        options["fused"] = fused
        super().__init__(params, options)

        for param_group in self.param_groups:
            if param_group["contiguous_params"]:
                param_list = param_group.contiguous_parameters
            else:
                param_list = param_group.parameters

            for param in param_list:
                assert param.is_leaf, "parameters must be leaf tensor"
                self.state[param] = dict()

                if param_group["fused"] and not param.is_cuda:
                    warnings.warn("Fused SGD only support cuda parameters.")
                    param_group["fused"] = False

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

    def _single_tensor_update(self, param_group):
        lr = param_group["lr"]
        l2 = param_group["weight_decay"]

        if param_group["contiguous_params"]:
            param_list = param_group.contiguous_parameters
        else:
            param_list = param_group.parameters

        for param in param_list:
            if param.grad is None:
                continue
            if param_group["momentum"] == 0.0:
                # TODO: Support param `maximize` in Naive SGD Optimizer. (zhengzekang)
                flow._C.dispatch_sgd_update(
                    self._sgd, (param, param.grad), learning_rate=lr, l2=l2
                )
            else:
                if "momentum_buf" not in self.state[param]:
                    self.state[param]["momentum_buf"] = flow.zeros_like(param)
                momentum_buf = self.state[param]["momentum_buf"]
                beta = param_group["momentum"]
                dampening = param_group["dampening"]
                nesterov = param_group["nesterov"]
                maximize = param_group["maximize"]
                flow._C.dispatch_momentum_update(
                    self._momentum_sgd,
                    (param, param.grad, momentum_buf),
                    learning_rate=lr,
                    l2=l2,
                    beta=beta,
                    dampening=dampening,
                    nesterov=nesterov,
                    maximize=maximize,
                )

    def _fused_update(self, param_group):
        use_momentum = param_group["momentum"] != 0
        param_list = []
        param_grad_list = []
        if use_momentum:
            momentum_buf_list = []

        for param in param_group.parameters:
            if param.grad is None:
                continue
            param_list.append(param)
            param_grad_list.append(param.grad)

            if use_momentum:
                if "momentum_buf" not in self.state[param]:
                    self.state[param]["momentum_buf"] = flow.zeros_like(param)
                momentum_buf_list.append(self.state[param]["momentum_buf"])

        if not use_momentum:
            flow._C.multi_tensor_sgd_update(
                model=param_list,
                model_diff=param_grad_list,
                scale=1.0,
                weight_decay=param_group["weight_decay"],
                learning_rate_val=param_group["lr"],
            )
        else:
            flow._C.multi_tensor_momentum_update(
                model=param_list,
                model_diff=param_grad_list,
                momentum_buf=momentum_buf_list,
                scale=1.0,
                weight_decay=param_group["weight_decay"],
                learning_rate_val=param_group["lr"],
                momentum=param_group["momentum"],
                dampening=param_group["dampening"],
                nesterov=param_group["nesterov"],
                maximize=param_group["maximize"],
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
                if param_group["fused"]:
                    self._fused_update(param_group)
                else:
                    self._single_tensor_update(param_group)

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
            beta = param_group["momentum"]
            l2 = param_group["weight_decay"]
            dampening = param_group["dampening"]
            nesterov = param_group["nesterov"]
            maximize = param_group["maximize"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            if beta == 0:
                optimizer_conf.naive_conf.SetInParent()
            else:
                optimizer_conf.momentum_conf.beta = beta
                # Only Momentum Optimizer support these params.
                optimizer_conf.momentum_conf.dampening = dampening
                optimizer_conf.momentum_conf.nesterov = nesterov
                optimizer_conf.momentum_conf.maximize = maximize

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
