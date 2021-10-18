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
    oneflow.diag,
    """
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
    """Returns the lower triangular part of a matrix (2-D tensor) or batch of matrices input along the specified diagonal, 
    the other elements of the result tensor out are set to 0.
    
    .. note::
        if diagonal = 0, the diagonal of the returned tensor will be the main diagonal,
        if diagonal > 0, the diagonal of the returned tensor will be above the main diagonal, 
        if diagonal < 0, the diagonal of the returned tensor will be below the main diagonal.

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
    """Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, 
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
    """The op computes the index with the largest value of a Tensor at specified axis.

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
    """The op computes the index with the largest value of a Tensor at specified axis.

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
    """Gather the element in batch dims. 
    
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
    oneflow._C.transpose,
    """Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

    The resulting out tensor shares its underlying storage with the input tensor, so changing the content of one would change the content of the other.

    Args:
        input (oneflow.Tensor): The input tensor.
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
