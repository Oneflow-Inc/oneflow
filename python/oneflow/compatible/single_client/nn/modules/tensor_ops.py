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


class TypeAs(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.to(dtype=target.dtype)


@register_tensor_op("type_as")
def type_as_op(input, target):
    """Returns this tensor cast to the type of the given tensor.
        This is a no-op if the tensor is already of the correct type.

    Args:
        input  (Tensor): the input tensor.
        target (Tensor): the tensor which has the desired type.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> target = flow.Tensor(np.random.randn(4, 5, 6), dtype = flow.int32)
        >>> input = input.type_as(target)
        >>> input.dtype
        oneflow.int32

    """
    return TypeAs()(input, target)


class Long(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.to(dtype=flow.int64)


@register_tensor_op("long")
def long_op(input):
    """`Tensor.long()` is equivalent to `Tensor.to(flow.int64)`. See to().

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64

    """
    return Long()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
