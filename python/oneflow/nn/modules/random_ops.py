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
        generator: (optional) – a pseudorandom number generator for sampling
        out (Tensor, optional) – the output tensor.

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
        # TODO: make shape process as a util
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, list, flow.Size)
        ), "shape should be int or tuple int!"
        self.device = device
        if isinstance(self.device, str):
            self.device = flow.device(self.device)
        self.requires_grad = requires_grad
        size = _single(size)
        if dtype is None:
            dtype = flow.float32
        if dtype not in [flow.float, flow.double]:
            raise NotImplementedError("Do not support such data type: {}".format(dtype))

        if generator is None:
            generator = flow.default_generator()
        self.generator = generator
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
            assert len(self.sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp
        self.size = size
        self.dtype = dtype

    def forward(self):
        if self.placement is not None:
            res = flow.F.consistent_rand(
                self.size, self.dtype, self.placement, self.sbp, self.generator
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
        generator (flow.Generator, optional) – a pseudorandom number generator for sampling
        device (torch.device, optional): The desired device of returned local tensor. If None, uses the
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
        >>> print(x)
        tensor([[0.5187, 0.4725, 0.974 ],
                [0.2193, 0.6767, 0.2337],
                [0.1863, 0.5853, 0.4277]], dtype=oneflow.float32)
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
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, list, flow.Size)
        ), "shape should be int or tuple int!"
        self.device = device
        if isinstance(self.device, str):
            self.device = flow.device(self.device)
        self.requires_grad = requires_grad
        size = _single(size)

        if generator is None:
            generator = flow.Generator()
        self.generator = generator
        self.placement = placement
        self.sbp = sbp
        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(self.sbp, flow.sbp.sbp):
                self.sbp = (self.sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
            assert len(self.sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp
        self.size = size
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
        generator (flow.Generator, optional) – a pseudorandom number generator for sampling
        device (torch.device, optional): The desired device of returned local tensor. If None, uses the
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
