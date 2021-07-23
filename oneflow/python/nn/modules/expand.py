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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional


class Expand(Module):
    def __init__(self, *sizes) -> None:
        super().__init__()
        self.expand_size = list(*sizes)

    def forward(self, x):
        if x.dtype == flow.int8:
            x = flow.experimental.cast(x, flow.int32)
        return flow.F.expand(x, self.expand_size)


@oneflow_export("expand")
@register_tensor_op("expand")
def expand_op(x, *sizes):
    """This operator expand the input tensor to a larger size.

    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions and the new ones will be appended at the front.

    For the new dimensions, the size cannot be set to -1.

    Args:
        x (oneflow.Tensor): The input Tensor.
        *sizes  (flow.Size or int): The desired expanded size.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[[[0, 1]],
        ...               [[2, 3]],
        ...               [[4, 5]]]]).astype(np.int32)

        >>> input = flow.Tensor(x)

        >>> out = input.expand(1, 3, 2, 2)
        >>> out.shape
        flow.Size([1, 3, 2, 2])

    """
    return Expand(sizes)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
