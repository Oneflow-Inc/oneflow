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
    oneflow.ones_like,
    """
    ones_like(input, *, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.ones_like.html.

    Returns a tensor filled with the scalar value 1, with the same size as input.
    flow.ones_like(input) is equivalent to flow.ones(input.shape, dtype=input.dtype)

    Args:
        input(Tensor): The size of input will determine size of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.random.rand(5), dtype=flow.float32)
        >>> y = flow.ones_like(x)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.zeros_like,
    """
    zeros_like(input, *, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.zeros_like.html.

    Returns a tensor filled with the scalar value 0, with the same size as input.
    flow.zeros_like(input) is equivalent to flow.zeros(input.shape, dtype=input.dtype)

    Args:
        input(Tensor): The size of input will determine size of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.random.rand(5), dtype=flow.float32)
        >>> y = flow.zeros_like(x)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.new_ones,
    """
    new_ones(x, size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.Tensor.new_ones.html.

    Returns a Tensor of size size filled with 1. By default, the returned Tensor has the same oneflow.dtype and oneflow.device as this tensor.

    Args:
        size (int...): a list, tuple, or flow.Size of integers defining the shape of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.Tensor(np.ones((1, 2, 3)))
        >>> y = x.new_ones((2, 2))
        >>> y
        tensor([[1., 1.],
                [1., 1.]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.empty,
    """
    empty(*size, *, dtype=None, device=None, placement=None, sbp=None, requires_grad=False, pin_memory=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.empty.html.

    Returns a tensor filled with uninitialized data.
    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        device (oneflow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp or List[flow.sbp], optional): The desired sbp of returned global tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
        pin_memory (bool, optional) – If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(4, 5)  # construct local empty tensor
        >>> y.shape
        oneflow.Size([4, 5])
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.empty(4, 5, placement=placement, sbp=flow.sbp.broadcast)  # construct consistent empty tensor
        >>> y.is_global
        True

    """,
)

add_docstr(
    oneflow.empty_like,
    """
    empty_like(input, *, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.empty_like.html.

    Returns an uninitialized tensor with the same size as :attr:`input`.
    ``oneflow.empty_like(input)`` is equivalent to
    ``oneflow.empty(input.size(), dtype=input.dtype, device=input.device)``.

    Args:
        input(Tensor): The size of input will determine size of the output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        device (oneflow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp or List[flow.sbp], optional): The desired sbp of returned global tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(2, 3)
        >>> y = flow.empty_like(x)
        >>> y.shape
        oneflow.Size([2, 3])

    """,
)
