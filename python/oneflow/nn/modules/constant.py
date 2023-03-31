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
from typing import List, Optional, Union
import numpy as np

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.common_types import _size_any_t
from oneflow.nn.modules.utils import _single, _handle_size_arg


class _ConstantBase:
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size],
        value: Union[float, int, complex],
        dtype: Optional[flow.dtype],
        device: Union[flow.device, int, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
        requires_grad: bool = False,
    ) -> None:
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, list, flow.Size)
        ), "shape should be int or tuple int!"
        self.device = device
        if isinstance(self.device, int):
            self.device = flow.device("cuda", self.device)
        if isinstance(self.device, str):
            self.device = flow.device(self.device)
        self.requires_grad = requires_grad
        size = _single(size)
        if dtype is None:
            dtype = flow.get_default_dtype()
        if placement is None:
            if device is None:
                self.device = flow.device("cpu")
        else:
            assert device is None
        self.placement = placement
        self.sbp = sbp
        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(self.sbp, flow.sbp.sbp):
                self.sbp = (self.sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
            assert len(self.sbp) == len(placement.ranks.shape)
        else:
            assert sbp is None, "sbp: %s" % sbp
        self.shape = size
        self.value = value
        self.dtype = dtype

    def forward(self):
        if self.placement is not None:
            if isinstance(self.value, flow.Tensor):
                assert (
                    self.value.ndim <= 1 and self.value.numel() == 1
                ), "Only tensor with single element or scalar tensor are supported as value!"
                res = flow._C.global_tensor_constant(
                    self.shape,
                    self.value,
                    dtype=self.dtype,
                    placement=self.placement,
                    sbp=self.sbp,
                )
            else:
                res = flow._C.global_constant(
                    self.shape,
                    self.value,
                    dtype=self.dtype,
                    placement=self.placement,
                    sbp=self.sbp,
                )
        else:
            if isinstance(self.value, flow.Tensor):
                assert (
                    self.value.ndim <= 1 and self.value.numel() == 1
                ), "Only tensor with single element or scalar tensor are supported as value!"
                res = flow._C.tensor_constant(
                    self.shape, self.value, dtype=self.dtype, device=self.device
                )
            else:
                res = flow._C.constant(
                    self.shape, self.value, dtype=self.dtype, device=self.device
                )
        res.requires_grad = self.requires_grad
        return res


def _handle_meta_args(
    input,
    size: Union[_size_any_t, List[int], flow.Size, None] = None,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    if isinstance(device, str):
        device = flow.device(device)
    if size is None:
        new_size = input.shape
    else:
        new_size = _handle_size_arg(size)
    if dtype is None:
        new_dtype = input.dtype
    else:
        new_dtype = dtype
    new_device = device
    new_placement = placement
    new_sbp = sbp
    new_requires_grad = requires_grad

    if new_device is not None:
        assert (
            new_placement is None
        ), "argument 'placement' must be None when argument 'device' exist"
        assert (
            new_sbp is None
        ), "argument 'sbp' must be None when argument 'device' exist"
    elif new_device is None and new_placement is None and new_sbp is None:
        new_device = input.device if input.is_local else None
        new_placement = input.placement if input.is_global else None
        new_sbp = input.sbp if input.is_global else None
    else:
        if new_placement is None and new_sbp is not None:
            assert (
                input.is_global
            ), "argument 'placement' must not be None when argument 'sbp' exist and Tensor is local"
            new_placement = input.placement
        elif new_placement is not None and new_sbp is None:
            assert (
                input.is_global
            ), "argument 'sbp' must not be None when argument 'placement' exist and Tensor is local"
            new_sbp = input.sbp
    assert isinstance(
        new_size, (int, tuple, list, flow.Size)
    ), f"argument 'size' must be tuple of ints, not %s" % (type(new_size))
    assert isinstance(
        new_dtype, flow.dtype
    ), f"argument 'dtype' must be flow.dtype, not %s" % (type(new_dtype))
    if new_placement is not None:
        assert isinstance(
            new_placement, flow.placement
        ), f"argument 'placement' must be flow.placement, not %s" % (
            type(new_placement)
        )
        assert isinstance(
            new_sbp, (flow.sbp.sbp, tuple)
        ), f"argument 'sbp' must be flow.sbp.sbp, not %s" % (type(new_sbp))
    else:
        assert isinstance(
            new_device, (str, flow.device)
        ), f"argument 'device' must be flow.device, not %s" % (type(new_device))
    assert isinstance(
        new_requires_grad, bool
    ), f"argument 'requires_grad' must be bool, not %s" % (type(new_requires_grad))

    return new_size, new_dtype, new_device, new_placement, new_sbp, new_requires_grad


class Ones(_ConstantBase):
    def __init__(
        self,
        size,
        dtype=None,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ):
        super().__init__(size, 1, dtype, device, placement, sbp, requires_grad)


def ones_op(
    *size: Union[_size_any_t, flow.Size, List[int]],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with the scalar value 1,
    with the shape defined by the variable argument `size`.

    Args:
        size (an integer or tuple of integer values): defining the shape of the output tensor. Can be \\
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional): the desired data type of returned tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.ones(5)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)
        >>> y = flow.ones(2,3) # construct local tensor
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.ones(4, 5, placement=placement, sbp=flow.sbp.broadcast) # construct global tensor
        >>> y.is_global
        True


    """
    size = _handle_size_arg(size)
    return Ones(size, dtype, device, placement, sbp, requires_grad).forward()


def ones_like_op(
    input,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(input, None, dtype, device, placement, sbp, requires_grad)
    return Ones(
        new_size, new_dtype, new_device, new_placement, new_sbp, new_requires_grad
    ).forward()


class Zeros(_ConstantBase):
    def __init__(
        self,
        size,
        dtype=None,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ):
        super().__init__(size, 0, dtype, device, placement, sbp, requires_grad)


def zeros_op(
    *size: Union[_size_any_t, flow.Size, List[int]],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with the scalar value 0,
    with the shape defined by the variable argument `size`.

    Args:
        size(an integer or tuple of integer values) - defining the shape of the output tensor. Can be \\
         a variable number of arguments or a collection like a list or tuple.
        dtype (flow.dtype, optional): the desired data type of returned tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.zeros(5)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)
        >>> y = flow.zeros(2,3)
        >>> y
        tensor([[0., 0., 0.],
                [0., 0., 0.]], dtype=oneflow.float32)

    """
    size = _handle_size_arg(size)
    return Zeros(size, dtype, device, placement, sbp, requires_grad).forward()


def zeros_like_op(
    input,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(input, None, dtype, device, placement, sbp, requires_grad)
    return Zeros(
        new_size, new_dtype, new_device, new_placement, new_sbp, new_requires_grad
    ).forward()


class Full(_ConstantBase):
    def __init__(
        self,
        size,
        value,
        dtype,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ):
        super().__init__(size, value, dtype, device, placement, sbp, requires_grad)


def full_op(
    size: Union[_size_any_t, flow.Size],
    fill_value: Union[float, int, complex],
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    """
    Creates a tensor of size `size` filled with fill_value. 
    The tensor’s dtype is inferred from `value`.

    Args:
        size(int...): a list, tuple, or oneflow.Size of integers defining the shape of the output tensor.
        fill_value(Scalar): the value to fill the output tensor with.
        dtype (oneflow.dtype, optional): the desired data type of returned tensor.
        device (oneflow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (oneflow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (oneflow.sbp.sbp or tuple of oneflow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.full((5,),5) 
        >>> y
        tensor([5, 5, 5, 5, 5], dtype=oneflow.int64)
        >>> y = flow.full((2,3),5.0) # construct local tensor
        >>> y
        tensor([[5., 5., 5.],
                [5., 5., 5.]], dtype=oneflow.float32)
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.full((2,3), 5.0, placement=placement, sbp=flow.sbp.broadcast)  # construct global tensor
        >>> y.is_global
        True

    """
    size = _handle_size_arg(size)
    if not isinstance(fill_value, (int, float, complex, flow.Tensor)):
        # handle numpy scalar dtype
        assert isinstance(
            fill_value.dtype, (np.dtype)
        ), "fill_value must be python scalar or numpy scalar."
        fill_value = fill_value.item()
    if dtype is None:
        dtype = flow.tensor(fill_value).dtype
    return Full(
        size, fill_value, dtype, device, placement, sbp, requires_grad
    ).forward()


def full_like_op(
    input,
    fill_value,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp], None
    ] = None,
    requires_grad: bool = False,
):
    """
    full_like(input, fill_value, \*, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor
    
    Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
    ``oneflow.full_like(input, fill_value)`` is equivalent to
    ``oneflow.full(input.size(), fill_value, dtype=input.dtype, device=input.device)``.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.full_like.html.

    Args:
        input(oneflow.Tensor)
        fill_value(Scalar): the value to fill the output tensor with.
        dtype (oneflow.dtype, optional): the desired data type of returned tensor.
        device (oneflow.device, optional): the desired device of returned tensor. Default: if None, uses the current device for the default tensor type
        placement (oneflow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (oneflow.sbp.sbp or tuple of oneflow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(2, 3)
        >>> y = flow.full_like(x, 2.0)
        >>> y
        tensor([[2., 2., 2.],
                [2., 2., 2.]], dtype=oneflow.float32)
        >>> y = flow.full_like(x, 2, dtype=flow.int32)
        >>> y
        tensor([[2, 2, 2],
                [2, 2, 2]], dtype=oneflow.int32)
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.full_like(x, 5.0, placement=placement, sbp=flow.sbp.broadcast)  # construct global tensor
        >>> y.is_global
        True

    """
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(input, None, dtype, device, placement, sbp, requires_grad)
    return Full(
        new_size,
        fill_value,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ).forward()


def new_ones_op(
    x, size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False
):
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(x, size, dtype, device, placement, sbp, requires_grad)
    if new_placement is not None:
        res = flow._C.global_constant(
            new_size, 1.0, dtype=new_dtype, placement=placement, sbp=sbp
        )
    else:
        res = flow._C.constant(new_size, 1.0, dtype=new_dtype, device=new_device)
    res.requires_grad = new_requires_grad
    return res


def new_zeros_op(
    x, size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False
):
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(x, size, dtype, device, placement, sbp, requires_grad)
    if new_placement is not None:
        res = flow._C.global_constant(
            new_size, 0.0, dtype=new_dtype, placement=new_placement, sbp=new_sbp
        )
    else:
        res = flow._C.constant(new_size, 0.0, dtype=new_dtype, device=new_device)
    res.requires_grad = new_requires_grad
    return res


def new_full_op(
    x,
    size,
    fill_value,
    dtype=None,
    device=None,
    placement=None,
    sbp=None,
    requires_grad=False,
):
    size = _handle_size_arg(size)
    (
        new_size,
        new_dtype,
        new_device,
        new_placement,
        new_sbp,
        new_requires_grad,
    ) = _handle_meta_args(x, size, dtype, device, placement, sbp, requires_grad)
    if flow.is_tensor(fill_value):
        assert (
            len(fill_value.size()) == 0
        ), "new_full(): argument 'fill_value' must be Number, not Tensor"
        fill_value = fill_value.item()

    if new_placement is not None:
        res = flow._C.global_constant(
            new_size, fill_value, dtype=new_dtype, placement=new_placement, sbp=new_sbp
        )
    else:
        res = flow._C.constant(new_size, fill_value, dtype=new_dtype, device=new_device)
    res.requires_grad = new_requires_grad
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
