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
from oneflow.framework.tensor import register_tensor_op


def atanh_op(input):
    """Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tanh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.array([0.5, 0.6, 0.7]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """
    return flow.F.atanh(input)


@register_tensor_op("atanh")
def atanh_op_tensor(input):
    """
    atanh() -> Tensor
    See :func:`oneflow.atanh`

    """
    return flow.F.atanh(input)


def arctanh_op(input):
    """

    Alias for :func:`oneflow.atanh`
    """
    return flow.F.atanh(input)


@register_tensor_op("arctanh")
def arctanh_op_tensor(input):
    """

    Alias for :func:`oneflow.atanh`
    """
    return flow.F.atanh(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
