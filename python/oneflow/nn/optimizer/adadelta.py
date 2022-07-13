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
from oneflow.nn.optimizer.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter


class Adadelta(Optimizer):
    r"""
    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        maximize: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert 1.0 >= rho >= 0.0, f"Invalid rho value: {rho}"

        options = dict()
        options["lr"] = lr
        options["rho"] = rho
        options["eps"] = eps
        options["maximize"] = maximize
        options["weight_decay"] = weight_decay
        super().__init__(params, options)

        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
                self._state[param]["square_avgs"] = flow.zeros_like(param)
                self._state[param]["acc_deltas"] = flow.zeros_like(param)

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
                loss = closure()
            for param_group in self.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "l2": param_group["weight_decay"],
                    "rho": param_group["rho"],
                    "epsilon": param_group["eps"],
                    "maximize": param_group["maximize"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    square_avgs_tensor = self._state[param]["square_avgs"]
                    acc_deltas_tensor = self._state[param]["acc_deltas"]
                    flow._C.dispatch_adadelta_update(
                        self._op,
                        (param, param.grad, square_avgs_tensor, acc_deltas_tensor),
                        **kwargs,
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
            l2 = param_group["weight_decay"]
            rho = param_group["rho"]
            epsilon = param_group["eps"]
            maximize = param_group["maximize"]

            optimizer_conf.base_learning_rate = lr
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
