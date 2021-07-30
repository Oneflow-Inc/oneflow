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
from typing import Optional, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.common_types import _size_any_t
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _single


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


def ones_op(
    size: Union[_size_any_t, flow.Size],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with the scalar value 1,
    with the shape defined by the variable argument `size`.

    Args:
        size (an integer or tuple of integer values) – defining the shape of the output tensor. Can be \\
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional) – the desired data type of returned tensor.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> y = flow.ones(5)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """
    return Ones(size, dtype, device, requires_grad)()


class Zeros(_ConstantBase):
    def __init__(self, size, dtype=None, device=None, requires_grad=False):
        super().__init__(size, 0, dtype, device, requires_grad)


def zeros_op(
    size: Union[_size_any_t, flow.Size],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with the scalar value 0,
    with the shape defined by the variable argument `size`.

    Args:
        size(an integer or tuple of integer values) - defining the shape of the output tensor. Can be \\
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional) – the desired data type of returned tensor.
        device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
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


def zeros_like_op(other):
    """
    Returns a tensor filled with the scalar value 0, with the same size as input.
    flow.zeros_like(input) is equivalent to flow.zeros(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client.experimental as flow
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


def ones_like_op(other):
    """
    Returns a tensor filled with the scalar value 1, with the same size as input.
    flow.ones_like(input) is equivalent to flow.ones(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.rand([5]))
        y = flow.ones_like(x)
        # [1. 1. 1. 1. 1. ]

    """
    return OnesLike()(other)


class NewOnes(Module):
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size] = None,
        dtype: Optional[flow.dtype] = None,
        device: Union[flow.device, str] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.device = device
        self.requires_grad = requires_grad
        if size != None:
            size = _single(size)
        self.size = size
        self.dtype = dtype

    def forward(self, x):
        new_size = self.size
        new_dtype = self.dtype
        new_device = self.device
        new_requires_grad = self.requires_grad
        if self.size is None:
            new_size = x.shape
        if self.dtype is None:
            new_dtype = x.dtype
        if self.device is None:
            new_device = x.device
        assert isinstance(
            new_size, (int, tuple, flow.Size)
        ), f"size parameter not correct, please check!"
        assert isinstance(
            new_dtype, flow.dtype
        ), f"dtype parameter not correct, please check!"
        assert isinstance(
            new_device, (str, flow.device)
        ), f"device parameter not correct, please check!"
        assert isinstance(
            new_requires_grad, bool
        ), f"requires_grad parameter not correct, please check!"
        res = flow.F.constant(new_size, 1.0, new_dtype)
        res = res.to(new_device)
        res.requires_grad = new_requires_grad
        return res


@register_tensor_op("new_ones")
def new_ones_op(x, size=None, dtype=None, device=None, requires_grad=False):
    """
    
    Returns a Tensor of size size filled with 1. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor.

    Args:
        size (int...): a list, tuple, or flow.Size of integers defining the shape of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.ones((1, 2, 3)))
        >>> y = x.new_ones((2, 2))
        >>> y
        tensor([[1., 1.],
                [1., 1.]], dtype=oneflow.float32)
    """
    return NewOnes(size=size, dtype=dtype, device=device, requires_grad=requires_grad)(
        x
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
