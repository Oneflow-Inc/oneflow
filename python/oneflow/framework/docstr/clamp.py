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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.clamp,
    """
    Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
    a resulting tensor:

    .. math::
        y_i = \\begin{cases}
            \\text{min} & \\text{if } x_i < \\text{min} \\\\
            x_i & \\text{if } \\text{min} \\leq x_i \\leq \\text{max} \\\\
            \\text{max} & \\text{if } x_i > \\text{max}
        \\end{cases}

    If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
    and :attr:`max` must be real numbers, otherwise they should be integers.

    Args:
        input (Tensor): the input tensor.
        min (Number): lower-bound of the range to be clamped to. Defaults to None.
        max (Number): upper-bound of the range to be clamped to. Defaults to None.
        out (Tensor, optional): the output tensor.

    For example:


    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=-0.5, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -0.5000, -0.3000], dtype=oneflow.float32)

        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=None, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -1.5000, -0.3000], dtype=oneflow.float32)

        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=-0.5, max=None)
        >>> output
        tensor([ 0.2000,  0.6000, -0.5000, -0.3000], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.clamp_min,
    """
    Clamp all elements in :attr:`input` which are less than :attr:`min` to :attr:`min` and return
    a resulting tensor:

    .. math::
        y_i = \max(min, x_i)

    If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
    must be real numbers, otherwise they should be integers.

    Args:
        input (Tensor): the input tensor.
        min (Number): lower-bound of the range to be clamped to.
        out (Tensor, optional): the output tensor.

    For example:


    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_min(input, min=-0.5)
        >>> output
        tensor([ 0.2000,  0.6000, -0.5000, -0.3000], dtype=oneflow.float32)

        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_min(input, min=-2)
        >>> output
        tensor([ 0.2000,  0.6000, -1.5000, -0.3000], dtype=oneflow.float32)

        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_min(input, min=1)
        >>> output
        tensor([1., 1., 1., 1.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.clamp_max,
    """
    Clamp all elements in :attr:`input` which are greater than :attr:`max` to :attr:`max` and return
    a resulting tensor:

    .. math::
        y_i = \min(max, x_i)

    If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`max`
    must be real numbers, otherwise they should be integers.

    Args:
        input (Tensor): the input tensor.
        max (Number): upper-bound of the range to be clamped to.
        out (Tensor, optional): the output tensor.

    For example:


    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_max(input, max=-0.5)
        >>> output
        tensor([-0.5000, -0.5000, -1.5000, -0.5000], dtype=oneflow.float32)

        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_max(input, max=-2)
        >>> output
        tensor([-2., -2., -2., -2.], dtype=oneflow.float32)

        >>> input = flow.Tensor([0.2, 0.6, -1.5, -0.3])
        >>> output = flow.clamp_max(input, max=1)
        >>> output
        tensor([ 0.2000,  0.6000, -1.5000, -0.3000], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.clip,
    """
    Alias for :func:`oneflow.clamp`. 
    """,
)
