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


def acosh_op(input):
    """Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

    .. math::

        \\text{out}_{i} = \\cosh^{-1}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> out1 = flow.acosh(x1)
        >>> out1
        tensor([1.3170, 1.7627, 2.0634], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([1.5, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.acosh(x2)
        >>> out2
        tensor([0.9624, 1.6094, 1.9827], device='cuda:0', dtype=oneflow.float32)

    """
    return flow.F.acosh(input)


@register_tensor_op("acosh")
def acosh_op_tensor(input):
    """

    acosh() -> Tensor

    See :func:`oneflow.acosh`

    """
    return flow.F.acosh(input)


def arccosh_op(input):
    """

    See :func:`oneflow.acosh`

    """
    return flow.F.acosh(input)


@register_tensor_op("arccosh")
def arccosh_op_tensor(input):
    """

    arccosh() -> Tensor

    See :func:`oneflow.acosh`

    """
    return flow.F.acosh(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
