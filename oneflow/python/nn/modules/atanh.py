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

from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module


class Atanh(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("atanh").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("atanh")
@register_tensor_op("atanh")
@experimental_api
def atanh_op(x):
    r"""Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \atanh^{-1}(\text{input}_{i})
    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def atanh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.atanh(x)


        x = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        out = atanh_Job(x)

        # out [0.54930615 0.6931472  0.8673005 ]

    """

    return Atanh()(x)