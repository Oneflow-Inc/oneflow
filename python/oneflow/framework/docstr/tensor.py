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
    oneflow.tensor,
    r"""
    Constructs a tensor with data, return a consistent tensor if placement and sbp are in kwargs,
       otherwise return a local tensor. 
       
    Arguments:
        data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar or tensor.
    Keyword Arguments:
        dtype (oneflow.dtype, optional) – the desired data type of returned tensor.
            Default: if None, infers data type from data.
        device (oneflow.device, optional): the desired device of returned tensor. If placement
            and sbp is None, uses the current cpu for the default tensor type.
        placement (oneflow.placement, optional): the desired placement of returned tensor.
        sbp (oneflow.sbp or tuple of oneflow.sbp, optional): the desired sbp of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False

    Note:
        The Keyword Argument device is mutually exclusive with placement and sbp.
        Consistent tensor only can be constructed from tensor.


    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.from_numpy,
    r"""
    Creates a ``Tensor`` from a ``numpy.ndarray``.

    The returned tensor and ndarray share the same memory. Modifications to the tensor
    will be reflected in the ndarray and vice versa.

    It currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16,
    numpy.int64, numpy.int32, numpy.int8, numpy.uint8.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.arange(6).reshape(2, 3)
        >>> t = flow.from_numpy(np_arr)
        >>> t
        tensor([[0, 1, 2],
                [3, 4, 5]], dtype=oneflow.int64)
        >>> np_arr[0, 0] = -1
        >>> t
        tensor([[-1,  1,  2],
                [ 3,  4,  5]], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.Tensor.atan2,
    r"""
    See :func:`oneflow.atan2`
    """,
)

add_docstr(
    oneflow.Tensor.expand_as,
    """
    expand_as(other) -> Tensor

    Expand this tensor to the same size as :attr:`other`.
    ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

    Please see :meth:`~Tensor.expand` for more information about ``expand``.

    Args:
        other (:class:`oneflow.Tensor`): The result tensor has the same size
            as :attr:`other`.
    """,
)

add_docstr(
    oneflow.Tensor.flatten,
    """
    See :func:`oneflow.flatten`
    """,
)

add_docstr(
    oneflow.Tensor.floor,
    """
    See :func:`oneflow.floor`
    """,
)

add_docstr(
    oneflow.Tensor.numel,
    """
    See :func:`oneflow.numel`
    """,
)

add_docstr(
    oneflow.Tensor.transpose,
    """
    See :func:`oneflow.transpose`
    """,
)

add_docstr(
    oneflow.Tensor.logical_not,
    """
    logical_not() -> Tensor
    See :func:`oneflow.logical_not`
    """,
)

add_docstr(
    oneflow.Tensor.std,
    """
    See :func:`oneflow.std`
    """,
)

add_docstr(
    oneflow.Tensor.var,
    """
    See :func:`oneflow.var`
    """,
)

add_docstr(
    oneflow.Tensor.squeeze,
    """
    See :func:`oneflow.squeeze`
    """,
)

add_docstr(
    oneflow.Tensor.unfold,
    """
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch.Tensor.unfold.

    Returns a view of the original tensor which contains all slices of `size` size from `self`
    tensor in the dimension `dimension`.

    Step between two slices is given by `step`.

    If sizedim is the size of dimension `dimension` for `self`, the size of dimension dimension in the
    returned tensor will be (sizedim - size) / step + 1.

    An additional dimension of size `size` is appended in the returned tensor.

    Args:
        dimension (int): dimension in which unfolding happens
        size (int): the size of each slice that is unfolded
        step (int): the step between each slice

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.arange(1., 8)
        >>> x
        tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> x.unfold(0, 2, 1)
        tensor([[ 1.,  2.],
                [ 2.,  3.],
                [ 3.,  4.],
                [ 4.,  5.],
                [ 5.,  6.],
                [ 6.,  7.]])
        >>> x.unfold(0, 2, 2)
        tensor([[ 1.,  2.],
                [ 3.,  4.],
                [ 5.,  6.]])
    """,
)

add_docstr(
    oneflow.Tensor.matmul,
    """
    See :func:`oneflow.matmul`
    """,
)

add_docstr(
    oneflow.Tensor.narrow,
    """
    See :func:`oneflow.narrow`
    """,
)

add_docstr(
    oneflow.Tensor.unsqueeze,
    """
    See :func:`oneflow.unsqueeze`
    """,
)

add_docstr(
    oneflow.Tensor.permute,
    """
    See :func:`oneflow.permute`
    """,
)

add_docstr(
    oneflow.Tensor.abs,
    """
    See :func:`oneflow.abs`
    """,
)

add_docstr(
    oneflow.Tensor.acos,
    """
    See :func:`oneflow.acos`
    """,
)

add_docstr(
    oneflow.Tensor.arccos,
    """
    See :func:`oneflow.arccos`
    """,
)

add_docstr(
    oneflow.Tensor.acosh,
    """
    See :func:`oneflow.acosh`
    """,
)

add_docstr(
    oneflow.Tensor.arccosh,
    """
    See :func:`oneflow.arccosh`
    """,
)

add_docstr(
    oneflow.Tensor.arctanh,
    """
    See :func:`oneflow.arctanh`
    """,
)

add_docstr(
    oneflow.Tensor.argmax,
    """
    See :func:`oneflow.argmax`
    """,
)

add_docstr(
    oneflow.Tensor.argmin,
    """
    See :func:`oneflow.argmin`
    """,
)

add_docstr(
    oneflow.Tensor.argsort,
    """This operator sorts the input Tensor at specified dim and return the indices of the sorted Tensor.

    Args:
        input (oneflow.Tensor): The input Tensor.
        dim (int, optional): dimension to be sorted. Defaults to the last dim (-1).
        descending (bool, optional): controls the sorting order (ascending or descending).

    Returns:
        oneflow.Tensor: The indices of the sorted Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[10, 2, 9, 3, 7],
        ...               [1, 9, 4, 3, 2]]).astype("float32")
        >>> input = flow.Tensor(x)
        >>> output = flow.argsort(input)
        >>> output
        tensor([[1, 3, 4, 2, 0],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, descending=True)
        >>> output
        tensor([[0, 2, 4, 3, 1],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, dim=0)
        >>> output
        tensor([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]], dtype=oneflow.int32)

    """,
)

add_docstr(
    oneflow.Tensor.atanh,
    """
    See :func:`oneflow.atanh`
    """,
)

add_docstr(
    oneflow.Tensor.backward,
    """
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward.

    Computes the gradient of current tensor w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If the tensor is non-scalar (i.e. its data has more than one element) and requires gradient, the function additionally requires specifying gradient. It should be a tensor of matching type and location, that contains the gradient of the differentiated function w.r.t. self.

    This function accumulates gradients in the leaves - you might need to zero .grad attributes or set them to None before calling it. See Default gradient layouts for details on the memory layout of accumulated gradients.

    Note:
        If you run any forward ops, create gradient, and/or call backward in a user-specified CUDA stream context, see Stream semantics of backward passes.
    Note:
        When inputs are provided and a given input is not a leaf, the current implementation will call its grad_fn (though it is not strictly needed to get this gradients). It is an implementation detail on which the user should not rely. See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    Args:
        gradient (Tensor or None): Gradient w.r.t. the tensor. If it is a tensor, it will be automatically converted to a Tensor that does not require grad unless create_graph is True. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable then this argument is optional.

        retain_graph (bool, optional): If False, the graph used to compute the grads will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

        create_graph (bool, optional): If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to False.
    """,
)

add_docstr(
    oneflow.Tensor.bmm,
    """
    See :func:`oneflow.bmm`
    """,
)

add_docstr(
    oneflow.Tensor.chunk,
    """
    See :func:`oneflow.chunk`
    """,
)

add_docstr(
    oneflow.Tensor.split,
    """
    See :func:`oneflow.split`
    """,
)

add_docstr(
    oneflow.Tensor.swapaxes,
    """
    See :func:`oneflow.swapaxes`
    """,
)

add_docstr(
    oneflow.Tensor.cast,
    """
    See :func:`oneflow.cast`
    """,
)

add_docstr(
    oneflow.Tensor.diag,
    """
    See :func:`oneflow.diag`
    """,
)

add_docstr(
    oneflow.Tensor.dim,
    """
    Tensor.dim() → int

    Returns the number of dimensions of self tensor.
    """,
)

add_docstr(
    oneflow.Tensor.element_size,
    """
    Tensor.element_size() → int

    Returns the size in bytes of an individual element.

    """,
)

add_docstr(
    oneflow.Tensor.exp,
    """
    See :func:`oneflow.exp`
    """,
)

add_docstr(
    oneflow.Tensor.erfinv,
    """
    See :func:`oneflow.erfinv`
    """,
)

add_docstr(
    oneflow.Tensor.erfinv_,
    """
    Inplace version of :func:`oneflow.erfinv`
    """,
)

add_docstr(
    oneflow.Tensor.fill_,
    """
    Tensor.fill_(value) → Tensor

    Fills self tensor with the specified value.
    """,
)

add_docstr(
    oneflow.Tensor.ge,
    """
    See :func:`oneflow.ge`
    """,
)

add_docstr(
    oneflow.Tensor.gelu,
    """
    See :func:`oneflow.gelu`
    """,
)

add_docstr(
    oneflow.Tensor.get_device,
    """
    Tensor.get_device() -> Device ordinal (Integer)

    For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides. For CPU tensors, an error is thrown.

  
    """,
)

add_docstr(
    oneflow.Tensor.gt,
    """
    See :func:`oneflow.gt`
    """,
)

add_docstr(
    oneflow.Tensor.log1p,
    """
    See :func:`oneflow.log1p`
    """,
)

add_docstr(
    oneflow.Tensor.mish,
    """
    See :func:`oneflow.mish`
    """,
)

add_docstr(
    oneflow.Tensor.mul,
    """
    See :func:`oneflow.mul`
    """,
)

add_docstr(
    oneflow.Tensor.mul_,
    """
    In-place version of :func`oneflow.Tensor.mul`.
    """,
)

add_docstr(
    oneflow.Tensor.negative,
    """
    See :func:`oneflow.negative`
    """,
)

add_docstr(
    oneflow.Tensor.nelement,
    """
    Tensor.nelement() → int

    Alias for numel()
    """,
)

add_docstr(
    oneflow.Tensor.floor_,
    r"""
    In-place version of :func:`oneflow.floor`

    """,
)

add_docstr(
    oneflow.Tensor.normal_,
    """
    normal_(mean=0, std=1, *, generator=None) -> Tensor

    Fills :attr:`self` tensor with elements samples from the normal distribution parameterized by :attr:`mean` and :attr:`std`.
    """,
)

add_docstr(
    oneflow.Tensor.numpy,
    """
    Tensor.numpy() → numpy.ndarray

    Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray share the same underlying storage. Changes to self tensor will be reflected in the ndarray and vice versa.
    """,
)

add_docstr(
    oneflow.Tensor.pow,
    """
    See :func:`oneflow.pow`
    """,
)

add_docstr(
    oneflow.Tensor.relu,
    """
    See :func:`oneflow.relu`
    """,
)

add_docstr(
    oneflow.Tensor.roll,
    """
    See :func:`oneflow.roll`
    """,
)

add_docstr(
    oneflow.Tensor.round,
    """
    See :func:`oneflow.round`
    """,
)

add_docstr(
    oneflow.Tensor.selu,
    """
    See :func:`oneflow.selu`
    """,
)

add_docstr(
    oneflow.Tensor.sigmoid,
    """
    See :func:`oneflow.sigmoid`
    """,
)

add_docstr(
    oneflow.Tensor.sign,
    """
    See :func:`oneflow.sign`
    """,
)

add_docstr(
    oneflow.Tensor.silu,
    """
    See :func:`oneflow.silu`
    """,
)

add_docstr(
    oneflow.Tensor.sinh,
    """
    See :func:`oneflow.sinh`
    """,
)

add_docstr(
    oneflow.Tensor.size,
    """
    The interface is consistent with PyTorch.
    
    Returns the size of the self tensor. If dim is not specified, the returned value is a torch.Size, a subclass of tuple. If dim is specified, returns an int holding the size of that dimension.

    Args:
        idx (int, optional): The dimension for which to retrieve the size.

  
    """,
)

add_docstr(
    oneflow.Tensor.softmax,
    """
    See :func:`oneflow.softmax`
    """,
)

add_docstr(
    oneflow.Tensor.softplus,
    """
    See :func:`oneflow.softplus`
    """,
)

add_docstr(
    oneflow.Tensor.softsign,
    """
    See :func:`oneflow.softsign`
    """,
)

add_docstr(
    oneflow.Tensor.tan,
    """
    See :func:`oneflow.tan`
    """,
)

add_docstr(
    oneflow.Tensor.tanh,
    """
    See :func:`oneflow.tanh`
    """,
)

add_docstr(
    oneflow.Tensor.tril,
    """
    See :func:`oneflow.tril`
    """,
)

add_docstr(
    oneflow.Tensor.triu,
    """
    See :func:`oneflow.triu`
    """,
)

add_docstr(
    oneflow.Tensor.uniform_,
    """
    Tensor.uniform_(from=0, to=1) → Tensor

    Fills self tensor with numbers sampled from the continuous uniform distribution:

    .. math::
        P(x)=1/(to-from)
    
    """,
)

add_docstr(
    oneflow.Tensor.copy_,
    """
    The interface is consistent with PyTorch.

    Tensor.copy_(src, non_blocking=False) → Tensor

    Copies the elements from src into self tensor and returns self.

    The src tensor must be broadcastable with the self tensor. It may be of a different data type or reside on a different device.

    Args:

        src (Tensor): the source tensor to copy from

        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.
    """,
)

add_docstr(
    oneflow.Tensor.to,
    """Performs Tensor dtype and/or device conversion.
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.

    .. note::
        If the ``input`` Tensor already
        has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
        Otherwise, the returned tensor is a copy of ``input`` with the desired.

    Args:
        input (oneflow.Tensor): An input tensor.
        *args (oneflow.Tensor or oneflow.device or oneflow.dtype): Positional arguments
        **kwargs (oneflow.device or oneflow.dtype) : Key-value arguments

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """,
)

add_docstr(
    oneflow.Tensor.gather,
    """
    oneflow.Tensor.gather(dim, index) -> Tensor

    See :func:`oneflow.gather`

    """,
)

add_docstr(
    oneflow.Tensor.clamp,
    """
    See :func:`oneflow.clamp`. 
    """,
)

add_docstr(
    oneflow.Tensor.clamp_,
    """
    Inplace version of :func:`oneflow.Tensor.clamp`. 
    """,
)

add_docstr(
    oneflow.Tensor.clip,
    """
    Alias for :func:`oneflow.Tensor.clamp`. 
    """,
)

add_docstr(
    oneflow.Tensor.clip_,
    """
    Alias for :func:`oneflow.Tensor.clamp_`. 
    """,
)
