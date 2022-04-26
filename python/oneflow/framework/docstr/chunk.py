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
    oneflow.chunk,
    """Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor. Last chunk will be bigger if the tensor size along the given dimension dim is not divisible by chunks.

    Args:
        input (oneflow.Tensor): The tensor to split.
        chunks (int): Number of chunks to return.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List of Tensors.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
               
        >>> arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.tensor(arr)
        >>> output = []
        >>> chunks = 3
        >>> output = flow.chunk(input, chunks=chunks, dim=2)
        >>> out_shape = []
        >>> for i in range(0, chunks):
        ...     out_shape.append(output[i].numpy().shape)
        >>> out_shape
        [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]

    """,
)
