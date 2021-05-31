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


class Sign(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("sign").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("sign")
@register_tensor_op("sign")
@experimental_api
def sign_op(x):
    """This operator computes the  sign of Tensor.

    The equation is: 

    .. math:: 

        out = sign(x)

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow.experimental as flow

        x = flow.Tensor(np.array([-2, 0, 2]).astype(np.float32))
        y = x.sign().numpy()

        # y [ -1  0  1 ]

    """
    return Sign()(x)
