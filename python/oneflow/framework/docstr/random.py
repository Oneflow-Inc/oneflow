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
    oneflow.bernoulli,
    """
    bernoulli(x, *, generator=None, out=None)
    
    This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.

    Args:
        x (Tensor): the input tensor of probability values for the Bernoulli distribution
        generator (Generator, optional): a pseudorandom number generator for sampling
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
        >>> x = flow.tensor(arr, dtype=flow.float32)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow._C.randn,
    """
    randn(*size, *, dtype=None, generator=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
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

    """,
)

add_docstr(
    oneflow._C.rand,
    """
    rand(*size, *, dtype=None, generator=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)

    The shape of the tensor is defined by the variable argument ``size``.

    Args:
        size (int... or oneflow.Size): Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.
        dtype (flow.dtype, optional): The desired data type of returned tensor. Default: ``flow.float32``.
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


    """,
)

add_docstr(
    oneflow._C.normal,
    r"""
    normal(mean, std, size, *, out=None, placement=None, sbp=None, generator=None, dtype=None, device=None, requires_grad=False) -> Tensor

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
    """,
)

add_docstr(
    oneflow._C.randint,
    """
    randint(low=0, high, size, *, dtype=None, generator=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    The shape of the tensor is defined by the variable argument ``size``.

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.randint.html.

    Args:
        low (int, optional):  Lowest integer to be drawn from the distribution. Default: 0.
        high (int):  One above the highest integer to be drawn from the distribution.
        size (tuple or oneflow.Size):  Defining the shape of the output tensor.
          Can be a variable number of arguments or a collection like a list or tuple or oneflow.Size.

    Keyword args:
        dtype (oneflow.dtype, optional): The desired data type of returned tensor. Default: ``flow.int64``.
        generator (oneflow.Generator, optional) – a pseudorandom number generator for sampling
        device (oneflow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (oneflow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (oneflow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
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

    """,
)

add_docstr(
    oneflow._C.randint_like,
    """
    randint_like(input, low=0, high, size, *, dtype=None, generator=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.randint_like.html.

    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    Args:
        input (oneflow.Tensor): the size of ``input`` will determine size of the output tensor.
        low (int, optional):  Lowest integer to be drawn from the distribution. Default: 0.
        high (int):  One above the highest integer to be drawn from the distribution.


    Keyword args:
        dtype (oneflow.dtype, optional): The desired data type of returned tensor. Default: ``flow.int64``.
        generator (oneflow.Generator, optional) – a pseudorandom number generator for sampling
        device (oneflow.device, optional): The desired device of returned local tensor. If None, uses the
          current device.
        placement (oneflow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (oneflow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> x = flow.randn(2, 2, generator=generator)
        >>> y = flow.randint_like(x, 0, 5, generator=generator) # construct local tensor
        >>> y
        tensor([[3, 4],
                [2, 4]], dtype=oneflow.int64)
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> y = flow.randint_like(x, 0, 5, generator=generator, placement=placement, sbp=flow.sbp.broadcast) # construct global tensor
        >>> y.is_global
        True

    """,
)

add_docstr(
    oneflow._C.randperm,
    r"""
    randperm(n, *, generator=None, dtype=torch.int64, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)

    Keyword args:
        generator(:class:`oneflow.Generator`, optional):  a pseudorandom number generator for sampling
        dtype (:class:`oneflow.dtype`, optional): the desired data type of returned tensor.
            Default: ``oneflow.int64``.
        device: the desired device of returned tensor. Default: cpu.
        placement:(:class:`flow.placement`, optional): The desired device of returned global tensor. If None,
            will construct local tensor.
        sbp: (:class:`flow.sbp`, optional): The desired sbp of returned global tensor. It must be equal with the
            numbers of placement.
        requires_grad(bool, optional): If autograd should record operations on the returned tensor. Default: False.

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

    """,
)
