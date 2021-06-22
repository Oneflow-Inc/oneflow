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
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op


class Acos(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.acos(x)


@oneflow_export("acos")
@register_tensor_op("acos")
@experimental_api
def acos_op(tensor):
    r"""
    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \text{out}_{i} = \arccos(\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> print(output.numpy())
        [1.0471976  0.9272952  0.79539883]

    """

    return Acos()(tensor)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
