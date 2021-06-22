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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.common_types import _size_any_t
from oneflow.python.nn.modules.utils import _single

from typing import Optional, Union


class _ConstantBase(Module):
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size],
        value: Union[float, int],
        dtype: Optional[flow.dtype],
        device: Union[flow.device, str] = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, flow.Size)
        ), "shape should be int or tuple int!"

        self.device = device
        self.requires_grad = requires_grad
        size = _single(size)
        if dtype is None:
            dtype = flow.float32

        if device is None:
            self.device = flow.device("cpu")

        self.shape = size
        self.value = value
        self.dtype = dtype

    def forward(self):
        res = flow.F.constant(self.shape, self.value, self.dtype)
        res = res.to(device=self.device)
        res.requires_grad = self.requires_grad
        return res


class Ones(_ConstantBase):
    def __init__(self, size, dtype=None, device=None, requires_grad=False):
        super().__init__(size, 1, dtype, device, requires_grad)


@oneflow_export("ones")
@experimental_api
def ones_op(
    size: Union[_size_any_t, flow.Size],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    requires_grad: bool = False,
):
    r"""
    Returns a tensor filled with the scalar value 1,
    with the shape defined by the variable argument `size`.

    Args:
        size (an integer or tuple of integer values) – defining the shape of the output tensor. Can be \
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional) – the desired data type of returned tensor.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> y = flow.ones(5)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """
    return Ones(size, dtype, device, requires_grad)()


class Zeros(_ConstantBase):
    def __init__(self, size, dtype=None, device=None, requires_grad=False):
        super().__init__(size, 0, dtype, device, requires_grad)


@oneflow_export("zeros")
@experimental_api
def zeros_op(
    size: Union[_size_any_t, flow.Size],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    requires_grad: bool = False,
):
    r"""
    Returns a tensor filled with the scalar value 0,
    with the shape defined by the variable argument `size`.

    Args:
        size(an integer or tuple of integer values) - defining the shape of the output tensor. Can be \
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional) – the desired data type of returned tensor.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> y = flow.zeros(5)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)

    """
    return Zeros(size, dtype, device, requires_grad)()


class ZerosLike(Module):
    def __init__(self):
        super().__init__()

    def forward(self, other):
        return flow.F.zeros_like(other)


@oneflow_export("zeros_like")
@experimental_api
def zeros_like_op(other):
    r"""
    Returns a tensor filled with the scalar value 0, with the same size as input.
    flow.zeros_like(input) is equivalent to flow.zeros(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.rand([5]))
        y = flow.zeros_like(x)
        # [0. 0. 0. 0. 0. ]

    """
    return ZerosLike()(other)


class OnesLike(Module):
    def __init__(self):
        super().__init__()

    def forward(self, other):
        return flow.F.ones_like(other)


@oneflow_export("ones_like")
@experimental_api
def ones_like_op(other):
    r"""
    Returns a tensor filled with the scalar value 1, with the same size as input.
    flow.ones_like(input) is equivalent to flow.ones(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.rand([5]))
        y = flow.ones_like(x)
        # [1. 1. 1. 1. 1. ]

    """
    return OnesLike()(other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
