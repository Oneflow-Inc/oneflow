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
    oneflow.diagonal,
    r"""
    oneflow.diagonal(input, offset, dim1, dim2) -> Tensor
    
    Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2 
    appended as a dimension at the end of the shape.
    
    Args:
        input (Tensor): the input tensor.Must be at least 2-dimensional.
        offset (Optional[int], 0): which diagonal to consider. Default: 0 (main diagonal)
        dim1 (Optional[int], 0): first dimension with respect to which to take diagonal. Default: 0
        dim2 (Optional[int], 1): second dimension with respect to which to take diagonal. Default: 1
    
    Returns:
        oneflow.Tensor: the output Tensor.

    For example:
    
    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randn(2,  3,  4)
        >>> output = flow.diagonal(input, offset=1, dim1=1, dim2=0)
        >>> output.shape
        oneflow.Size([4, 1])
    """,
)

add_docstr(
    oneflow.diag,
    r"""diag(input, diagonal)
    If input is a vector (1-D tensor), then returns a 2-D square tensor with the elements of input as the diagonal.
    If input is a matrix (2-D tensor), then returns a 1-D tensor with diagonal elements of input.

    Args:
        input (Tensor): the input tensor.
        diagonal (Optional[int], 0): The diagonal to consider. 
            If diagonal = 0, it is the main diagonal. If diagonal > 0, it is above the main diagonal. If diagonal < 0, it is below the main diagonal. Defaults to 0.
    
    Returns:
        oneflow.Tensor: the output Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array(
        ...     [
        ...        [1.0, 2.0, 3.0],
        ...        [4.0, 5.0, 6.0],
        ...        [7.0, 8.0, 9.0],
        ...     ]
        ... )

        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> flow.diag(input)
        tensor([1., 5., 9.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.tril,
    r"""tril(input, diagonal)
    Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input along the specified diagonal, 
    the other elements of the result tensor out are set to 0.
    
    .. note::
        - if diagonal = 0, the diagonal of the returned tensor will be the main diagonal,
        - if diagonal > 0, the diagonal of the returned tensor will be above the main diagonal, 
        - if diagonal < 0, the diagonal of the returned tensor will be below the main diagonal.

    Args:
        input (Tensor): the input tensor. 
        diagonal (int, optional): the diagonal to specify. 

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.ones(shape=(3, 3)).astype(np.float32))
        >>> flow.tril(x)
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.triu,
    r"""Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, 
    the other elements of the result tensor out are set to 0.
    
    Args:
        input (Tensor): the input tensor. 
        diagonal (int, optional): the diagonal to consider

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.tensor(np.ones(shape=(3, 3)).astype(np.float32))
        >>> flow.triu(x)
        tensor([[1., 1., 1.],
                [0., 1., 1.],
                [0., 0., 1.]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.argmax,
    r"""The op computes the index with the largest value of a Tensor at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        keepdim (bool optional):  whether the output tensor has dim retained or not. Ignored if dim=None.

    Returns:
        oneflow.Tensor: A Tensor(dtype=int64) contains the index with the largest value of `input`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[1, 3, 8, 7, 2],
        ...            [1, 9, 4, 3, 2]], dtype=flow.float32)
        >>> output = flow.argmax(input)
        >>> output
        tensor(6, dtype=oneflow.int64)
        >>> output = flow.argmax(input, dim=1)
        >>> output
        tensor([2, 1], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.argmin,
    r"""The op computes the index with the largest value of a Tensor at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        keepdim (bool optional):  whether the output tensor has dim retained or not. Ignored if dim=None.

    Returns:
        oneflow.Tensor: A Tensor(dtype=int64) contains the index with the largest value of `input`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[4, 3, 1, 0, 2],
        ...            [5, 9, 7, 6, 8]], dtype=flow.float32)
        >>> output = flow.argmin(input)
        >>> output
        tensor(3, dtype=oneflow.int64)
        >>> output = flow.argmin(input, dim=1)
        >>> output
        tensor([3, 0], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.batch_gather,
    r"""Gather the element in batch dims. 
    
    Args:
        in (Tensor): the input tensor. 
        indices (Tensor): the indices tensor, its dtype must be int32/64. 

    For example:

    Example 1: 

    .. code-block:: python

        >>> import oneflow as flow 
        >>> import numpy as np 

        >>> x = flow.Tensor(np.array([[1, 2, 3], 
        ...                           [4, 5, 6]]))
        >>> indices = flow.tensor(np.array([1, 0]).astype(np.int64))
        >>> out = flow.batch_gather(x, indices)

        tensor([[4., 5., 6.],
                [1., 2., 3.]], dtype=oneflow.float32)

    Example 2: 

    .. code-block:: python

        >>> import oneflow as flow 
        >>> import numpy as np 

        >>> x = flow.Tensor(np.array([[[1, 2, 3], [4, 5, 6]], 
        ...                           [[1, 2, 3], [4, 5, 6]]]))
        >>> indices = flow.tensor(np.array([[1, 0], 
        ...                                 [0, 1]]).astype(np.int64))
        >>> out = flow.batch_gather(x, indices)

        tensor([[[4., 5., 6.],
                 [1., 2., 3.]],
                [[1., 2., 3.],
                 [4., 5., 6.]]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.transpose,
    r"""transpose(input, dim0, dim1)
    Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

    The resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.

    Args:
        input (oneflow.Tensor): the input tensor.
        dim0 (int): the first dimension to be transposed.
        dim1 (int): the second dimension to be transposed.
    Returns:
        Tensor: A transposed tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> out = flow.transpose(input, 0, 1).shape
        >>> out
        oneflow.Size([6, 2, 5, 3])

    """,
)

add_docstr(
    oneflow.stack,
    r"""stack(inputs, dim)
    Concatenates a sequence of tensors along a new dimension.
    The returned tensor shares the same underlying data with input tensors.

    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1]`
    can be used. Negative :attr:`dim` will correspond to :meth:`stack`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.

    Args:
        inputs (List[oneflow.Tensor]): the list of input tensors. Each tensor should have the same shape.
        dim (int): the index at which to insert the concatenated dimension.

    Returns:
        A `Tensor`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x1 = flow.tensor(np.random.rand(1, 3, 5))
        >>> x2 = flow.tensor(np.random.rand(1, 3, 5))
        >>> y = flow.stack([x1, x2], dim = -1)
        >>> y.shape
        oneflow.Size([1, 3, 5, 2])
    """,
)

add_docstr(
    oneflow.squeeze,
    r"""squeeze(input, dim)
    This operator removes the specified dimention which size is 1 of the input Tensor.
    If the `dim` is not specified, this operator will remove all the dimention which size is 1 of the input Tensor.

    The amount of element in return value is the same as Tensor `input`.

    Args:
        input (oneflow.Tensor): the input Tensor.
        dim (int, optinal): Defaults to None, if given, the input will be squeezed only in this dimension.

    Returns:
        Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([[[[1, 1, 1]]]]).astype(np.int32))
        >>> input.shape
        oneflow.Size([1, 1, 1, 3])
        >>> out = flow.squeeze(input, dim=[1, 2]).shape
        >>> out
        oneflow.Size([1, 3])

    """,
)

add_docstr(
    oneflow.cat,
    r"""
    cat(inputs, dim=0) -> Tensor 
    Alias for :func:`oneflow.concat`.
    """,
)

add_docstr(
    oneflow.concat,
    r"""
    concat(inputs, dim=0) -> Tensor

    Concatenate two or more `Tensor` s at specified dim.

    Analogous to `numpy.concatenate <https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html>`_

    Args:
        inputs: a `list` of `Tensor`
        dim: a `int`.

    Returns:
        A `Tensor`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input1 = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input2 = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input3 = flow.tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)

        >>> out = flow.cat([input1, input2, input3], dim=1) # equal to using flow.concat()
        >>> out.shape
        oneflow.Size([2, 18, 5, 3])

    """,
)

add_docstr(
    oneflow.gather,
    """
    oneflow.gather(input, dim, index, sparse_grad=False) -> Tensor
    
    Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by::

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :attr:`input` and :attr:`index` must have the same number of dimensions.
    It is also required that ``index.size(d) <= input.size(d)`` for all
    dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to gather

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = np.random.randn(3, 4, 3, 5)
        >>> index = np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5))
        >>> output = flow.gather(flow.Tensor(input), 1, flow.tensor(index, dtype=flow.int64))
        >>> output.shape
        oneflow.Size([3, 4, 3, 5])

    """,
)

add_docstr(
    oneflow.gather_nd,
    r"""
    oneflow.gather_nd(input, index) -> Tensor
    
    This operator is a high-dimensional extension of `gather`, `index` is a K-dimensional
    tensor, which is regarded as a index of input Tensor `input`.

    Each element defines a slice of `input`:

    .. math::

        output[i_{0},i_{1},...,i_{K-2}] = input[index(i_{0},i_{1},...,i_{K-2})]


    Args:
        input: The input Tensor.
        index: The slice indices.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([[1, 2,3], [4, 5,6],[7,8,9]]), dtype=flow.float)
        >>> index_1 = flow.tensor(np.array([[0], [2]]), dtype=flow.int)
        >>> out_1 = flow.gather_nd(input,index_1)
        >>> print(out_1.shape)
        oneflow.Size([2, 3])
        >>> out_1
        tensor([[1., 2., 3.],
                [7., 8., 9.]], dtype=oneflow.float32)
        >>> index_2 = flow.tensor(np.array([[0,2], [2,1]]), dtype=flow.int)
        >>> out_2 = flow.gather_nd(input,index_2)
        >>> out_2
        tensor([3., 8.], dtype=oneflow.float32)

    """,
)
