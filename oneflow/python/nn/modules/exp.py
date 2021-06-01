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
import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class Exp(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("exp").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("exp")
@register_tensor_op("exp")
@experimental_api
def exp_op(x):
    """This operator computes the exponential of Tensor.

    The equation is: 

    .. math:: 

        out = e^x

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow

        x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        y = x.exp().numpy()

        # y [ 2.7182817  7.389056  20.085537 ]

    """
    return Exp()(x)
