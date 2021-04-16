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

from collections import OrderedDict
from typing import List, Dict, Callable, Union

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.parameter import Parameter
from oneflow.python.framework.tensor import Tensor


class Optimizer(object):
    def __init__(self):
        # self._parameters = OrderedDict()
        self._parameters = list()
        self._default_state = OrderedDict()
        self._default_state["step"] = 0
        self._op = None

    def add_param_group(self, param_group) -> None:
        TODO()

    def load_state_dict(self, state_dict) -> None:
        TODO()

    def state_dict(self):
        TODO()

    def step(self, closure: Union[Callable, None] = None) -> Union[Tensor, None]:
        raise NotImplementedError()

    def zero_grad(self, set_to_none: bool = False):
        for param in self._parameters:
            if set_to_none:
                param.grad = None
            else:
                param.grad.fill_(0)
                # param.grad.zeros_()


@oneflow_export("optim.SGD")
class SGD(Optimizer):
    r"""
    TODO
    """

    def __init__(
        self,
        parameters: Union[List[Parameter], List[Dict]],
        lr: float,
        scale: float = 1.0,
        momentum: float = 0.0,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        self._default_state["lr"] = lr
        self._default_state["scale"] = scale
        # self._default_state["momentum"] = momentum # not use now

        self._op = (
            flow.builtin_op("sgd_update")
            .Input("model")
            .Input("model_diff")
            .Input("learning_rate")
            .Attr("scale", self._default_state["scale"])
            .Attr("weight_decay", 0.0)
            .Attr("l1", 0.0)
            .Attr("l2", 0.0)
            .Build()
        )

        for param in parameters:
            if isinstance(param, Parameter):
                self._parameters.append(param)
            else:  # Dict
                assert "param" in param
                self._parameters.append(param["param"])
                if any(x in param for x in ["lr", "momentum"]):
                    raise NotImplementedError(
                        "Not support specifying per-parameter options now"
                    )

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        lr_tensor = flow.Tensor([self._default_state["lr"]])
        for param in self._parameters:
            if param.grad is not None:
                self._op(param, param.grad, lr_tensor)
        self._default_state["step"] = self._default_state["step"] + 1
        return loss
