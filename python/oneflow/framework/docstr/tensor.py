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

        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be accumulated into .grad. All other Tensors will be ignored. If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the attr::tensors.   
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
    oneflow.Tensor.fill_,
    """
    Tensor.fill_(value) → Tensor

    Fills self tensor with the specified value.
    """,
)

add_docstr(
    oneflow.Tensor.ge,
    """
    Computes  

    .. math::
        \t { input } \geq \t { other }
    
    element-wise.

    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (Tensor): the tensor to compare

        other (Tensor or float): the tensor or value to compare

    Keyword Arguments:
        out (Tensor, optional): the output tensor.
    
    Returns:
        A boolean tensor that is True where input is greater than or equal to other and False elsewhere
    
   
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
    oneflow.Tensor.normal_,
    """
    Tensor.normal_(mean=0, std=1, *, generator=None) → Tensor

    Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
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
        dim (int, optional): The dimension for which to retrieve the size.

  
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
    The interface is consistent with PyTorch.

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.triu.html#torch.triu.

    Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and above the main diagonal are retained. A positive value excludes just as many diagonals above the main diagonal, and similarly a negative value includes just as many diagonals below the main diagonal.
    
    Args:
        input (Tensor): the input tensor.

        diagonal (int, optional): the diagonal to consider

    Keyword Arguments:
        out (Tensor, optional): the output tensor.


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
    oneflow.Tensor.xavier_uniform_,
    """
    Args:


        gain: an optional scaling factor


    
    """,
)

add_docstr(
    oneflow.Tensor.xavier_normal_,
    """
    Args:


        gain: an optional scaling factor


    """,
)

add_docstr(
    oneflow.Tensor.kaiming_normal_,
    """
    Args:

        a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')

        mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

        nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).


    """,
)

add_docstr(
    oneflow.Tensor.kaiming_uniform_,
    """
    Args:



        a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')

        mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

        nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).

    """,
)

add_docstr(
    oneflow.Tensor.copy_,
    """
    The interface is consistent with PyTorch.

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_.

    Tensor.copy_(src, non_blocking=False) → Tensor

    Copies the elements from src into self tensor and returns self.

    The src tensor must be broadcastable with the self tensor. It may be of a different data type or reside on a different device.

    Args:

        src (Tensor): the source tensor to copy from

        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.
    """,
)



