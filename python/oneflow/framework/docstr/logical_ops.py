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
    oneflow.logical_and,
    """
    Computes the element-wise logical AND of the given input tensors.
    Zeros are treated as False and nonzeros are treated as True.

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute AND with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input1 = flow.tensor(np.array([1, 0, 1]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 1, 0]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.logical_and(input1, input2)
        >>> out
        tensor([ True, False, False], dtype=oneflow.bool)

    """,
)


add_docstr(
    oneflow.logical_or,
    """
    Computes the element-wise logical OR of the given input tensors. 
    Zeros are treated as False and nonzeros are treated as True.

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute OR with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 0, 1]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 0, 0]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.logical_or(input1, input2)
        >>> out
        tensor([ True, False,  True], dtype=oneflow.bool)

    """,
)


add_docstr(
    oneflow.logical_xor,
    """
    Computes the element-wise logical XOR of the given input tensors. 
    Zeros are treated as False and nonzeros are treated as True.

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute XOR with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python
    
        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 0, 1]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 0, 0]).astype(np.float32), dtype=flow.float32)
        >>> out = flow.logical_xor(input1, input2)
        >>> out
        tensor([False, False,  True], dtype=oneflow.bool)

    """,
)

add_docstr(
    oneflow.logical_not,
    r"""
    Computes the element-wise logical NOT of the given input tensors.
    Zeros are treated as False and nonzeros are treated as True.
    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute NOT with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 0, -1], dtype=flow.float32)
        >>> out = flow.logical_not(input)
        >>> out
        tensor([False,  True, False], dtype=oneflow.bool)

    """,
)
