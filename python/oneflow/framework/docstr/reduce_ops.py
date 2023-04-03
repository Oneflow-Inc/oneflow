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
    oneflow.max,
    """
    oneflow.max(input, dim=None, keepdim=False)

    Computes the maximum value of all elements in the input tensor.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor or Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int64)): If :attr:`dim` is `None`, returns 
        the maximum value of all elements in the `input` tensor. Otherwise, returns a tuple of Tensor (values, indices), 
        where the `values` are the maximum value of all elements in the `input` tensor,
        the `indices` are the indices of the elements in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.max(input)
        tensor(6., dtype=oneflow.float32)
        >>> result = flow.max(input, dim=1)
        >>> result.values
        tensor([5., 6.], dtype=oneflow.float32)
        >>> result.indices
        tensor([2, 1], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.min,
    """
    oneflow.min(input, dim=None, keepdim=False)
    
    Computes the minimum value of all elements in the input tensor.
    
    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    Returns:
        Tensor or Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int64)): If :attr:`dim` is `None`, returns 
        the minimum value of all elements in the `input` tensor. Otherwise, returns a tuple of Tensor (values, indices), 
        where the `values` are the minimum value of all elements in the `input` tensor,
        the `indices` are the indices of the elements in the original input tensor.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[4, 1, 5], [2, 6, 3]])
        >>> flow.min(input)
        tensor(1., dtype=oneflow.float32)
        >>> result = flow.min(input, dim=1)
        >>> result.values
        tensor([1., 2.], dtype=oneflow.float32)
        >>> result.indices
        tensor([1, 0], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.sum,
    """
    oneflow.sum(input, dim=None, keepdim=False) -> Tensor

    Computes the sum of row of elements in a tensor in the given dimension. If the dimension is None, sum of all elements will be caculated.
    
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed :func:`oneflow.squeeze()`, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s). 

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int or tuple of ints, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.sum(input)
        tensor(21., dtype=oneflow.float32)
        >>> flow.sum(input, dim=0)
        tensor([5., 7., 9.], dtype=oneflow.float32)
        >>> flow.sum(input, dim=1)
        tensor([ 6., 15.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.mean,
    """
    oneflow.mean(input, dim=None, keepdim=False) -> Tensor
    
    Computes the mean of row of elements in a tensor in the given dimension. If the dimension is None, mean of all elements will be caculated.
    
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed :func:`oneflow.squeeze()`, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s). 

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int or tuple of ints, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.mean(input)
        tensor(3.5000, dtype=oneflow.float32)
        >>> flow.mean(input, dim=0)
        tensor([2.5000, 3.5000, 4.5000], dtype=oneflow.float32)
        >>> flow.mean(input, dim=1)
        tensor([2., 5.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.prod,
    """
    oneflow.prod(input, dim=None, keepdim=False) -> Tensor

    Computes the product of row of elements in a tensor in the given dimension. If the dimension is None, product of all elements will be caculated.
    
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed :func:`oneflow.squeeze()`, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s). 

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int or tuple of ints, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.prod(input)
        tensor(720., dtype=oneflow.float32)
        >>> flow.prod(input, dim=0)
        tensor([ 4., 10., 18.], dtype=oneflow.float32)
        >>> flow.prod(input, dim=1)
        tensor([  6., 120.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.all,
    """
    oneflow.all(input, dim=None, keepdim=False) -> Tensor

    For each row of `input` in the given dimension `dim`, returns True if all element in the row evaluate to True and False otherwise. If the dimension is None, compute if all elements in the input tensor to true.
    
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed :func:`oneflow.squeeze()`, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s). 

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]]) < 4
        >>> input
        tensor([[ True,  True,  True],
                [False, False, False]], dtype=oneflow.bool)
        >>> flow.all(input)
        tensor(False, dtype=oneflow.bool)
        >>> flow.all(input, 1)
        tensor([ True, False], dtype=oneflow.bool)
        >>> flow.all(input, 1, True)
        tensor([[ True],
                [False]], dtype=oneflow.bool)
    """,
)

add_docstr(
    oneflow.any,
    """
    oneflow.any(input, dim=None, keepdim=False) -> Tensor
    
    For each row of `input` in the given dimension `dim`, returns True if any element in the row evaluate to True and False otherwise. If the dimension is None, compute if any elements in the input tensor to true.
    
    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim` is squeezed :func:`oneflow.squeeze()`, resulting in the output tensor having 1 (or `len(dim)`) fewer dimension(s). 

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: `None`
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]]) < 4
        >>> input
        tensor([[ True,  True,  True],
                [False, False, False]], dtype=oneflow.bool)
        >>> flow.any(input)
        tensor(True, dtype=oneflow.bool)
        >>> flow.any(input, 0)
        tensor([True, True, True], dtype=oneflow.bool)
        >>> flow.any(input, 0, True)
        tensor([[True, True, True]], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.nansum,
    r"""oneflow.nansum(input, dim, keepdim=False, *, dtype=None) -> Tensor

    Returns the sum of each row of the ``input`` tensor in the given dimension ``dim``,
    treating Not a Numbers (NaNs) as zero. If ``dim`` is a list of dimensions, 
    reduce over all of them.

    If ``keepdim`` is ``True``, the output tensor is of the same size as ``input`` except 
    in the dimension(s) ``dim`` where it is of size 1. 
    Otherwise, ``dim`` is squeezed (see :class:`oneflow.squeeze()`), 
    resulting in the output tensor having 1 (or ``len(dim)``) fewer dimension(s).

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nansum.html.

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int, optional): the dimension to reduce. Default: ``None``
        keepdim (bool, optional): whether the output tensor has ``dim`` retained or not. Default: `False`
        dtype (oneflow.dtype, optional): the desired data type of returned tensor. 
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: ``None``.

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1., 2., float("nan")])
        >>> flow.nansum(x)
        tensor(3., dtype=oneflow.float32)
        >>> x = flow.tensor([[1., float("nan")], [float("nan"), 2]])
        >>> flow.nansum(x, dim=1)
        tensor([1., 2.], dtype=oneflow.float32)
        >>> x = flow.tensor([float("nan") for i in range(3)])
        >>> flow.nansum(x)
        tensor(0., dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.logsumexp,
    r"""
    oneflow.logsumexp(input, dim, keepdim=False) -> Tensor
    
    Returns the log of summed exponentials of each row of the :attr:`input`
    tensor in the given dimension :attr:`dim`. The computation is numerically
    stabilized.

    For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{logsumexp}(x)_{{i}} = \log \sum_j \exp(x_{{ij}})

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.12/generated/torch.logsumexp.html.

    Args:
        input (oneflow.Tensor): the Input Tensor
        dim (int or tuple of ints): the dimension or dimensions to reduce.
        keepdim (bool, optional): whether the output tensor has dim retained or not. Default: `False`

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> flow.logsumexp(input, 0)
        tensor([4.0486, 5.0486, 6.0486], dtype=oneflow.float32)
        >>> flow.logsumexp(input, 1)
        tensor([3.4076, 6.4076], dtype=oneflow.float32)

    """,
)
