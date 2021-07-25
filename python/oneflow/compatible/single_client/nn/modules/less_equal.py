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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class LessEqual(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if x.dtype != flow.float32:
            x = flow.experimental.cast(x, flow.float32)
        if isinstance(y, int) or isinstance(y, float):
            y = flow.Tensor(
                [float(y)], dtype=flow.float32, device=flow.device(x.device.type)
            )
        if y.dtype != flow.float32:
            y = flow.experimental.cast(y, flow.float32)
        return flow.F.broadcast_less_equal(x, y)


@register_tensor_op("le")
def less_equal_op(x, y):
    """Returns the truth value of :math:`x <= y` element-wise.

    Args:
        x (oneflow.compatible.single_client.Tensor): A Tensor
        y (oneflow.compatible.single_client.Tensor): A Tensor

    Returns:
        oneflow.compatible.single_client.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> input1 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    """
    return LessEqual()(x, y)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
