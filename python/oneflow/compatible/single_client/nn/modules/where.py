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


class Where(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, condition, x, y):
        assert condition.dtype == flow.int32 or condition.dtype == flow.int8
        if isinstance(x, int) or isinstance(x, float):
            x = flow.Tensor(
                [float(x)],
                dtype=flow.float32,
                device=flow.device(condition.device.type),
            )
        if isinstance(y, int) or isinstance(y, float):
            y = flow.Tensor(
                [float(y)],
                dtype=flow.float32,
                device=flow.device(condition.device.type),
            )
        assert (
            condition.device.type == x.device.type
            and condition.device.type == y.device.type
        )
        assert len(condition.shape) == len(x.shape) and len(condition.shape) == len(
            y.shape
        ), f"The dim of where module's inputs can not match, please check!"
        broadcast_cond = condition
        broadcast_x = x
        broadcast_y = y
        broadcast_like_shape = []
        broadcast_condition_axes = []
        broadcast_x_axes = []
        broadcast_y_axes = []
        for i in range(len(x.shape)):
            max_dim = max(x.shape[i], max(y.shape[i], condition.shape[i]))
            broadcast_like_shape.append(max_dim)
            if max_dim != condition.shape[i]:
                broadcast_condition_axes.append(i)
            if max_dim != x.shape[i]:
                broadcast_x_axes.append(i)
            if max_dim != y.shape[i]:
                broadcast_y_axes.append(i)
        broadcast_like_tensor = flow.experimental.zeros(
            tuple(broadcast_like_shape), dtype=flow.float32
        )
        broadcast_like_tensor = broadcast_like_tensor.to(x.device.type)
        broadcast_like_tensor.requires_grad = x.requires_grad or y.requires_grad
        if len(broadcast_condition_axes) != 0:
            condition = flow.experimental.cast(condition, flow.float32)
            broadcast_cond = flow.experimental.broadcast_like(
                condition, broadcast_like_tensor, tuple(broadcast_condition_axes)
            )
            broadcast_cond = flow.experimental.cast(broadcast_cond, flow.int32)
        if len(broadcast_x_axes) != 0:
            broadcast_x = flow.experimental.broadcast_like(
                x, broadcast_like_tensor, broadcast_axes=tuple(broadcast_x_axes)
            )
        if len(broadcast_y_axes) != 0:
            broadcast_y = flow.experimental.broadcast_like(
                y, broadcast_like_tensor, broadcast_axes=tuple(broadcast_y_axes)
            )
        return flow.F.where(broadcast_cond, broadcast_x, broadcast_y)


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
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

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
    return Where()(condition, x, y)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
