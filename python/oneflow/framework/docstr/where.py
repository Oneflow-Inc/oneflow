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
    oneflow.where,
    """Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.
    If the element in condition is larger than 0,

    it will take the `x` element, else it will take the `y` element

    .. note::
        If :attr:`x` is None and :attr:`y` is None,  flow.where(condition) is 
        identical to flow.nonzero(condition, as_tuple=True).
        
        The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be broadcastable.

    Args:
        condition (IntTensor): When 1 (nonzero), yield x, otherwise yield y
        x (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                            where :attr:`condition` is True
        y (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                            where :attr:`condition` is False
    Returns:
        Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = flow.tensor(
        ...    np.array([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        ...    dtype=flow.float32,
        ... )
        >>> y = flow.tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        >>> condition = flow.tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        >>> out = condition.where(x, y)
        >>> out #doctest: +ELLIPSIS
        tensor([[1.0000, 0.3139],
                ...
                [0.0478, 1.0000]], dtype=oneflow.float32)

    """,
)
