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


@register_tensor_op("where")
def where_op(condition, x, y):
    """Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.
    If the element in condition is larger than 0,

    it will take the `x` element, else it will take the `y` element

    .. note::

        The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be broadcastable.
        It will take the `x` element, else it will take the `y` element.

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
        >>> x = flow.Tensor(
        ...    np.array([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        ...    dtype=flow.float32,
        ... )
        >>> y = flow.Tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        >>> condition = flow.Tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        >>> out = condition.where(x, y)
        >>> out #doctest: +ELLIPSIS
        tensor([[1.    , 0.3139],
                ...
                [0.0478, 1.    ]], dtype=oneflow.float32)

    """

    assert condition.dtype == flow.int32 or condition.dtype == flow.int8
    if isinstance(x, int) or isinstance(x, float):
        x = flow.Tensor(
            [float(x)], dtype=flow.float32, device=flow.device(condition.device.type),
        )
    if isinstance(y, int) or isinstance(y, float):
        y = flow.Tensor(
            [float(y)], dtype=flow.float32, device=flow.device(condition.device.type),
        )
    assert (
        condition.device.type == x.device.type
        and condition.device.type == y.device.type
    )
    assert len(condition.shape) == len(x.shape) and len(condition.shape) == len(
        y.shape
    ), f"The dim of where module's inputs can not match, please check!"
    return flow.F.where(condition, x, y)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
