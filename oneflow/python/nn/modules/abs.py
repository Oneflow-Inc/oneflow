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


class Abs(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return flow.F.abs(x)


@oneflow_export("abs")
@register_tensor_op("abs")
@experimental_api
def abs_op(x):
    r"""Return the absolute value of each element in input tensor:math:`y = |x|` element-wise.

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)

    """
    return Abs()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
