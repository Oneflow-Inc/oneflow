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

from typing import List, Dict, Callable, Union, Any, Iterator, Tuple
from types import GeneratorType

import numpy as np
import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.parameter import Parameter
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.optimizer.optimizer import ParamGroup
from oneflow.python.nn.optimizer.optimizer import Optimizer


@oneflow_export("optim.Adam")
class Adam(Optimizer):
    r"""
    TODO
    """

    def __init__(
        self,
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert (
            betas[0] >= 0.0 and betas[0] < 1.0
        ), f"Invalid beta parameter at index 0: {betas[0]}"
        assert (
            betas[1] >= 0.0 and betas[1] < 1.0
        ), f"Invalid beta parameter at index 1: {betas[1]}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert scale > 0.0, f"Invalid scale factor: {scale}"

        self._default_options = dict()
        self._default_options["lr"] = lr
        self._default_options["eps"] = eps
        self._default_options["beta"] = betas
        self._default_options["weight_decay"] = weight_decay
        self._default_options["amsgrad"] = amsgrad
        self._default_options["scale"] = scale

        # Add parameters
        if isinstance(parameters, GeneratorType):
            self._param_groups.append(ParamGroup(parameters, self._default_options))
        else:  # List[Dict]
            for param in parameters:
                self._param_groups.append(ParamGroup(param, self._default_options))

        for param_group in self._param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
                self._state[param]["exp_avg"] = flow.tmp.zeros(
                    # TODO: zeros module support flow.Size parameter
                    tuple(param.shape)
                )
                self._state[param]["exp_avg_sq"] = flow.tmp.zeros(
                    # TODO: zeros module support flow.Size parameter
                    tuple(param.shape)
                )

        self._op = (
            flow.builtin_op("adam_update")
            .Input("model")
            .Input("model_diff")
            .Input("learning_rate")
            .Input("m")
            .Input("v")
            .Attr("scale", self._default_options["scale"])
            .Attr("l1", 0.0)
            .Attr("l2", 0.0)
            .Attr("beta1", 0.9)
            .Attr("beta2", 0.999)
            .Attr("epsilon", 1e-8)
            .Attr("weight_decay", 0.0)
            .Build()
        )

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self._param_groups:
            lr_tensor = flow.Tensor([param_group.options["lr"]])
            for param in param_group.parameters:
                if param.grad is None:
                    continue
                m_tensor = self._state[param]["exp_avg"]
                v_tensor = self._state[param]["exp_avg_sq"]
                self._op(param, param.grad, lr_tensor, m_tensor, v_tensor)

        self._state["step"] = self._state["step"] + 1

        return loss
