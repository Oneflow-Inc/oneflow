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
    bernoulli(input, p, *, generator=None, out=None)
    
    This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.

    Args:
        input (Tensor): the input tensor of probability values for the Bernoulli distribution
        p (float, optional): the probability for the Bernoulli distribution. If specified, Bernoulli distribution will use p for sampling, not input
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
        >>> y = flow.bernoulli(x, 1)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)
        >>> y = flow.bernoulli(x, p=0)
        >>> y
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]], dtype=oneflow.float32)

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
    oneflow._C.randn_like,
    """
    randn_like(input, *, dtype=None, generator=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a tensor with the same size as `input` that is filled with random numbers from a normal distribution with mean 0 and variance 1.
    flow.randn_like(input) is equivalent to flow.randn(input.size(), dtype=input.dtype, device=input.device).

    Args:
        input (oneflow.Tensor): the size of ``input`` will determine size of the output tensor.
        dtype (flow.dtype, optional): The desired data type of returned tensor. defaults to the dtype of `input`.
        generator (flow.Generator, optional): a pseudorandom number generator for sampling
        device (flow.device, optional): The desired device of returned local tensor. If None, defaults to the device of `input`.
        placement (flow.placement, optional): The desired device of returned global tensor. If None, will
          construct local tensor.
        sbp (flow.sbp, optional): The desired sbp of returned global tensor. It must be equal with the
          numbers of placement, If None, will construct local tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3,3) # construct local tensor
        >>> y = flow.randn_like(x)
        >>> y.shape
        oneflow.Size([3, 3])
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", ranks=[0])
        >>> sbp = flow.sbp.broadcast
        >>> z = flow.randn_like(y, placement=placement, sbp=sbp) # construct global tensor
        >>> z.is_global
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
        >>> generator.manual_seed(0) #doctest: +ELLIPSIS
        <oneflow._oneflow_internal.Generator object at ...>
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
        >>> generator.manual_seed(0) #doctest: +ELLIPSIS
        <oneflow._oneflow_internal.Generator object at ...>
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

    Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

    The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.randint_like.html.

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
        >>> generator.manual_seed(0) #doctest: +ELLIPSIS
        <oneflow._oneflow_internal.Generator object at ...>
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
        >>> generator.manual_seed(0) #doctest: +ELLIPSIS
        <oneflow._oneflow_internal.Generator object at ...>
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

add_docstr(
    oneflow.multinomial,
    """
    multinomial(input, num_samples, replacement=False, generator=None) -> LongTensor
    
    Returns a tensor where each row contains :attr:`num_samples` indices sampled
    from the multinomial probability distribution located in the corresponding row
    of tensor :attr:`input`.

    .. note::
      The rows of :attr:`input` do not need to sum to one (in which case we use
      the values as weights), but must be non-negative, finite and have
      a non-zero sum.

    Indices are ordered from left to right according to when each was sampled
    (first samples are placed in first column).

    If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

    If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
    :math:`(m x num\_samples)`.

    If replacement is ``True``, samples are drawn with replacement.

    If not, they are drawn without replacement, which means that when a
    sample index is drawn for a row, it cannot be drawn again for that row.

    .. note::
        When drawn without replacement, :attr:`num_samples` must be lower than
        number of non-zero elements in :attr:`input` (or the min number of non-zero
        elements in each row of :attr:`input` if it is a matrix).

    Args:
        input (Tensor): the input tensor containing probabilities
        num_samples (int): number of samples to draw
        replacement (bool, optional): whether to draw with replacement or not

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> gen = flow.manual_seed(0)
        >>> weights = flow.tensor([0, 10, 3, 0], dtype=flow.float) # create a tensor of weights
        >>> flow.multinomial(weights, 2)
        tensor([1, 2], dtype=oneflow.int64)
        >>> flow.multinomial(weights, 4, replacement=True)
        tensor([1, 2, 1, 1], dtype=oneflow.int64)

    """,
)
