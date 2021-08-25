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

import oneflow as flow
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _single


def bernoulli(input, *, generator=None, out=None):
    """This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.

    Args:
        input(Tensor) - the input tensor of probability values for the Bernoulli distribution
        generator: (optional): a pseudorandom number generator for sampling
        out (Tensor, optional): the output tensor.

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> arr = np.array(
        ...    [
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)


    """
    return flow.F.bernoulli(input, flow.float32, generator)


def _rand_op_common_process(
    size, device=None, generator=None, placement=None, sbp=None
):
    assert size is not None, "shape must not be None!"
    assert isinstance(
        size, (int, tuple, list, flow.Size)
    ), "shape should be int or tuple int!"
    if isinstance(device, str):
        device = flow.device(device)
    size = _single(size)
    processed_sbp = sbp
    if generator is None:
        generator = flow.Generator()
    if placement is not None:
        assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
        if isinstance(processed_sbp, flow.sbp.sbp):
            processed_sbp = (processed_sbp,)
        else:
            for elem in sbp:
                assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
        assert len(processed_sbp) == len(placement.hierarchy)
    else:
        assert sbp is None, "sbp: %s" % sbp
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
            res = flow.F.consistent_rand(
                self.size, self.placement, self.sbp, self.dtype, self.generator
            )
        else:
            res = flow.F.rand(self.size, self.dtype, self.device, self.generator)
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
        size (int... or flow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or flow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned consistent tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned consistent tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.rand(3,3)
        >>> x.shape
        flow.Size([3, 3])
        >>> x.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.rand(3, 3, placement=placement, sbp=sbp)
        >>> x.is_consistent
        True

    """
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if generator is None:
        generator = flow.default_generator()
    return Rand(size, generator, dtype, layout, device, placement, sbp, requires_grad)()


class RandN(Module):
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
            res = flow.F.consistent_randn(
                self.size, self.placement, self.sbp, self.dtype, self.generator
            )
        else:
            res = flow.F.randn(self.size, self.dtype, self.device, self.generator)
        res.requires_grad = self.requires_grad
        return res


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
        size (int... or flow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or flow.Size.
        out (optional): The output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
        layout (optional): The desired layout of returned Tensor.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (flow.placement, optional): The desired device of returned consistent tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned consistent tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3,3)
        >>> x.shape
        flow.Size([3, 3])
        >>> x.is_consistent
        False
        >>> placement = flow.placement("cpu", {0:[0]})
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.randn(3,3,placement=placement,sbp=sbp)
        >>> x.is_consistent
        True

    """
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if generator is None:
        generator = flow.default_generator()
    return RandN(
        size, generator, dtype, layout, device, placement, sbp, requires_grad
    )()


class Randperm(Module):
    def __init__(
        self,
        n,
        generator: flow.Generator = None,
        dtype: flow.dtype = flow.int32,
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
        self.requires_grad = requires_grad
        (
            self.size,
            self.device,
            self.generator,
            self.placement,
            self.sbp,
        ) = _rand_op_common_process(1, device, generator, placement, sbp)
        self.dtype = dtype

    def forward(self, out=None):
        if self.placement is not None:
            res = flow.F.consistent_randperm(
                self.n, self.placement, self.sbp, self.generator
            )
        else:
            res = flow.F.randperm(self.n, self.device, self.generator)
        res.requires_grad = self.requires_grad
        return res.to(dtype=self.dtype)


def randperm(
    n: flow.int32,
    generator: flow.Generator = None,
    out=None,
    dtype: flow.dtype = flow.int32,
    layout=None,
    device: Union[flow.device, str, None] = None,
    placement: flow.placement = None,
    sbp: flow._oneflow_internal.sbp.sbp = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
):
    r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)
    
    Keyword args:
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        out (Tensor, optional): output Tensor,not supported yet.
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int32``.
        layout: layout is not supported yet.
        device: the desired device of returned tensor. Default: cpu.
        placement:(:class:`flow.placement`, optional): The desired device of returned consistent tensor. If None,
            will construct local tensor.
        sbp: (:class:`flow.sbp`, optional): The desired sbp of returned consistent tensor. It must be equal with the
            numbers of placement.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.
        pin_memory(bool, optional):pin_memory is not supported yet.

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> flow.randperm(5, generator=generator)
        tensor([2, 4, 3, 0, 1], dtype=oneflow.int32)
    """
    assert out is None, "out not supported yet"
    assert layout is None, "layout not supported yet"
    if generator is None:
        generator = flow.default_generator()
    return Randperm(
        n, generator, dtype, layout, device, placement, sbp, requires_grad, pin_memory
    )(out)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
