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
from typing import Optional, Union, List

import oneflow as flow
from oneflow.nn.module import Module
from oneflow.nn.common_types import _size_any_t
from oneflow.nn.modules.utils import _single, _handle_size_arg


def _rand_op_common_process(
    size, device=None, generator=None, placement=None, sbp=None
):
    if isinstance(device, str):
        device = flow.device(device)
    size = _single(size)
    processed_sbp = sbp
    if placement is not None:
        if isinstance(processed_sbp, flow.sbp.sbp):
            processed_sbp = (processed_sbp,)
    return size, device, generator, placement, processed_sbp


class Rand(Module):
    def __init__(
        self,
        size,
        generator=None,
        dtype=None,
        layout=None,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ) -> None:
        super().__init__()
        self.requires_grad = requires_grad
        (
            self.size,
            self.device,
            self.generator,
            self.placement,
            self.sbp,
        ) = _rand_op_common_process(size, device, generator, placement, sbp)
        self.dtype = dtype

    def forward(self):
        if self.placement is not None:
            res = flow._C.rand(
                self.size,
                placement=self.placement,
                sbp=self.sbp,
                dtype=self.dtype,
                generator=self.generator,
            )
        else:
            res = flow._C.rand(
                self.size,
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
        res.requires_grad = self.requires_grad
        return res


def rand_op(
    *size,
    out=None,
    generator=None,
    dtype: Optional[flow.dtype] = None,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False
):
    """
    Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.rand(3,3) # construct local tensor
        >>> x.shape
        oneflow.Size([3, 3])
        >>> x.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.rand(3, 3, placement=placement, sbp=sbp) # construct global tensor
        >>> x.is_global
        True


    """
    size = _handle_size_arg(size)
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if placement is not None:
        return flow._C.rand(
            size=size,
            placement=placement,
            sbp=sbp,
            dtype=dtype,
            generator=generator,
            requires_grad=requires_grad,
        )
    else:
        return flow._C.rand(
            size=size,
            dtype=dtype,
            device=device,
            generator=generator,
            requires_grad=requires_grad,
        )


def randn_op(
    *size,
    out=None,
    generator=None,
    dtype: Optional[flow.dtype] = None,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False
):
    """
    Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3,3) # construct local tensor
        >>> x.shape
        oneflow.Size([3, 3])
        >>> x.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.randn(3,3,placement=placement,sbp=sbp) # construct global tensor
        >>> x.is_global
        True

    """
    size = _handle_size_arg(size)
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if placement is not None:
        return flow._C.randn(
            size=size,
            placement=placement,
            sbp=sbp,
            dtype=dtype,
            generator=generator,
            requires_grad=requires_grad,
        )
    else:
        return flow._C.randn(
            size=size,
            dtype=dtype,
            device=device,
            generator=generator,
            requires_grad=requires_grad,
        )


class RandInt(Module):
    def __init__(
        self,
        low: flow.int64,
        high: flow.int64,
        size: tuple,
        generator: flow.Generator = None,
        dtype: Optional[flow.dtype] = None,
        device=None,
        placement=None,
        sbp=None,
        requires_grad=False,
    ) -> None:
        super().__init__()

        assert low < high
        self.requires_grad = requires_grad
        (
            self.size,
            self.device,
            self.generator,
            self.placement,
            self.sbp,
        ) = _rand_op_common_process(size, device, generator, placement, sbp)
        self.dtype = dtype
        self.low = low
        self.high = high

    def forward(self):
        if self.placement is not None:
            res = flow._C.randint(
                self.low,
                self.high,
                size=self.size,
                placement=self.placement,
                sbp_tuple=self.sbp,
                dtype=self.dtype,
                generator=self.generator,
                requires_grad=self.requires_grad,
            )
        else:
            res = flow._C.randint(
                self.low,
                self.high,
                size=self.size,
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
                requires_grad=self.requires_grad,
            )
        return res


def randint_op(
    low: flow.int64,
    high: flow.int64,
    size: tuple,
    out=None,
    generator=None,
    dtype: Optional[flow.dtype] = None,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False,
):
    """
    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.int64``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional) – a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> y = flow.randint(0, 5, (3,3), generator=generator) # construct local tensor
        >>> y
        tensor([[2, 2, 3],
                [4, 3, 4],
                [2, 4, 2]], dtype=oneflow.int64)
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.randint(0, 5, (3,3), generator=generator, placement=placement, sbp=flow.sbp.broadcast) # construct global tensor
        >>> y.is_global
        True

    """
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if placement is not None:
        return flow._C.randint(
            low,
            high,
            size=size,
            generator=generator,
            dtype=dtype,
            placement=placement,
            sbp_tuple=sbp,
            requires_grad=requires_grad,
        )
    else:
        return flow._C.randint(
            low,
            high,
            size=size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )


class RandPerm(Module):
    def __init__(
        self,
        n,
        generator: flow.Generator = None,
        dtype: Optional[flow.dtype] = None,
        layout=None,
        device: Union[flow.device, str, None] = None,
        placement: flow.placement = None,
        sbp: flow._oneflow_internal.sbp.sbp = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        assert n >= 0
        self.n = n
        self.dtype = dtype
        (
            _,
            self.device,
            self.generator,
            self.placement,
            self.sbp,
        ) = _rand_op_common_process((), device, generator, placement, sbp)
        self.requires_grad = requires_grad

    def forward(self, out=None):
        if self.placement is not None:
            res = flow._C.randperm(
                self.n,
                placement=self.placement,
                sbp=self.sbp,
                generator=self.generator,
                requires_grad=self.requires_grad,
            )
        else:
            res = flow._C.randperm(
                self.n,
                device=self.device,
                generator=self.generator,
                requires_grad=self.requires_grad,
            )
        return res.to(dtype=self.dtype)


def randperm_op(
    n: flow.int64,
    generator: flow.Generator = None,
    out=None,
    dtype: Optional[flow.dtype] = None,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> flow.Tensor:
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)

    Keyword args:
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        out (Tensor, optional): output Tensor,not supported yet.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        layout: layout is not supported yet.
        device: the desired device of returned tensor. Default: cpu.
        placement:(:class:`flow.placement`, optional): The desired device of returned global tensor. If None,
            will construct local tensor.
        sbp: (:class:`flow.sbp`, optional): The desired sbp of returned global tensor. It must be equal with the
            numbers of placement.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.
        pin_memory(bool, optional):pin_memory is not supported yet.

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> y = flow.randperm(5, generator=generator) # construct local tensor
        >>> y
        tensor([2, 4, 3, 0, 1], dtype=oneflow.int64)
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.randperm(5, generator=generator, placement=placement, sbp=flow.sbp.broadcast) # construct global tensor
        >>> y.is_global
        True

    """
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    assert pin_memory is False, "pin_memory not supported yet"
    if dtype is None:
        dtype = flow.int64
    if placement is not None:
        return flow._C.randperm(
            n=n,
            placement=placement,
            sbp=sbp,
            generator=generator,
            requires_grad=requires_grad,
        ).to(dtype)
    else:
        return flow._C.randperm(
            n=n, device=device, generator=generator, requires_grad=requires_grad
        ).to(dtype)


def normal_op(
    mean,
    std,
    *size: Union[_size_any_t, flow.Size, List[int]],
    out=None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    generator=None,
    dtype: Optional[flow.dtype] = None,
    device: Union[flow.device, str, None] = None,
    requires_grad: bool = False
):
    r"""
    Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.

    Args:
        mean (float):  the mean for all distributions
        std (float):  the standard deviation for all distributions
        size (int...):  a sequence of integers defining the shape of the output tensor.

    Keyword args:
        out (Tensor, optional):  the output tensor.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: `oneflow.float32`.
        device: the desired device of returned tensor. Default: cpu.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> y = flow.normal(0, 1, 5, generator=generator)
        >>> y
        tensor([2.2122, 1.1631, 0.7740, 0.4838, 1.0434], dtype=oneflow.float32)
    """
    if len(size) == 1:
        size = size[0]
        if isinstance(size, int):
            size = (size,)

    if placement is not None:
        return flow._C.normal(
            mean=mean,
            std=std,
            size=size,
            out=out,
            placement=placement,
            sbp=sbp,
            dtype=dtype,
            generator=generator,
            requires_grad=requires_grad,
        )
    else:
        return flow._C.normal(
            mean=mean,
            std=std,
            size=size,
            out=out,
            dtype=dtype,
            device=device,
            generator=generator,
            requires_grad=requires_grad,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
