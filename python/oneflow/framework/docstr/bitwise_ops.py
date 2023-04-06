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
    oneflow.bitwise_and,
    """
    Computes the bitwise AND of input and other.
    The input tensor must be of integral or Boolean types.
    For bool tensors, it computes the logical AND.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.bitwise_and.html

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute bitwise AND with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> flow.bitwise_and(x, 2)
        tensor([0, 2, 2], dtype=oneflow.int64)
        >>> y = flow.tensor([5, 6, 7])
        >>> flow.bitwise_and(x, y)
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)


add_docstr(
    oneflow.bitwise_or,
    """
    Computes the bitwise OR of input and other.
    The input tensor must be of integral or Boolean types.
    For bool tensors, it computes the logical OR.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.bitwise_or.html

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute OR with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> flow.bitwise_or(x, 4)
        tensor([5, 6, 7], dtype=oneflow.int64)
        >>> y = flow.tensor([5, 6, 7])
        >>> flow.bitwise_or(x, y)
        tensor([5, 6, 7], dtype=oneflow.int64)

    """,
)


add_docstr(
    oneflow.bitwise_xor,
    """
    Computes the bitwise XOR of input and other.
    The input tensor must be of integral or Boolean types.
    For bool tensors, it computes the logical XOR.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.bitwise_xor.html

    Args:
        input (oneflow.Tensor): The input Tensor
        other (oneflow.Tensor): The Tensor to compute XOR with

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> flow.bitwise_xor(x, 2)
        tensor([3, 0, 1], dtype=oneflow.int64)
        >>> y = flow.tensor([5, 6, 7])
        >>> flow.bitwise_xor(x, y)
        tensor([4, 4, 4], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.bitwise_not,
    """
    Computes the bitwise NOT of input.
    The input tensor must be of integral or Boolean types.
    For bool tensors, it computes the logical NOT.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.bitwise_not.html

    Args:
        input (oneflow.Tensor): The input Tensor

    Returns:
        oneflow.Tensor: The output Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> flow.bitwise_not(x)
        tensor([-2, -3, -4], dtype=oneflow.int64)
        >>> x = flow.tensor([0, 0, 1]).bool()
        >>> flow.bitwise_not(x)
        tensor([ True,  True, False], dtype=oneflow.bool)

    """,
)
