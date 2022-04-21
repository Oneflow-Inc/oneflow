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
    oneflow.tensordot,
    r"""
    tensordot(a, b, dims=Union[int, Tensor, Tuple[List[int]], List[List[int]]]) -> Tensor
    
    Compute tensor dot along given dimensions.
    
    Given two tensors a and b, and dims which has two list containing dim indices, `tensordot` traverses the two 
    lists and calculate the tensor dot along every dim pair.

    Args:
        a (oneflow.Tensor): The input tensor to compute tensordot
        b (oneflow.Tensor): The input tensor to compute tensordot
        dims (int or list or tuple or oneflow.Tensor):
            The dims to calculate tensordot.
            If it's an integer or oneflow.Tensor with only one element,
            the last `dims` of tensor a and the first `dims` of tensor b will be calculated.
            If it's a list or tuple or oneflow.Tensor,
            it must contain two array-like object, which represent the dims of tensor a and tensor b to be calculated.
        
    Returns:
        oneflow.Tensor: The result tensor

    For example:
    
    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.randn(3, 4, 5)
        >>> b = flow.randn(4, 5, 6)
        >>> flow.tensordot(a, b, dims=2).shape
        oneflow.Size([3, 6])
        >>> b = flow.randn(5, 6, 7)
        >>> flow.tensordot(a, b, dims=1).shape
        oneflow.Size([12, 42])
        >>> b = flow.randn(3, 4, 7)
        >>> flow.tensordot(a, b, dims=[[0, 1], [0, 1]]).shape
        oneflow.Size([5, 7])
    
    Note:
        The operation is equivalent to the series of operations:

        - Permute the dimensions of the tensor A that require tensordot to the end

        - Permute the dimensions of the tensor B that require tensordot to the start

        - Reshape the permuted tensor A into a 2-dimensional tensor, where the size of the 0th dimension is the product of the dimensions that do not require dot product, and the size of the 1st dimension is the product of the dimensions that require dot product

        - Reshape the permuted tensor B into a 2-dimensional tensor, where the size of the 0th dimension is the product of the dimensions that require dot product, and the size of the 1st dimension is the product of the dimensions that do not require dot product

        - Calculate the matrix multiplication of reshaped tensor A and reshaped tensor B

        - Reshape the result of matrix multiplication, the target shape is the concatenation of the dimensions that do not require tensordot of tensor A and B

    This series of operations can be equivalently represented by the following code:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.randn(2, 4, 3)
        >>> b = flow.randn(3, 4, 2)
        >>> dims = [[0, 2], [2, 0]]
        >>> permuted_a = a.permute(1, 0, 2) # 0, 2 are the dimensions requiring tensordot and are placed in the end in permuting
        >>> permuted_b = b.permute(2, 0, 1) # 2, 0 are the dimensions requiring tensordot and are placed at the beginning in permuting
        >>> reshaped_a = permuted_a.reshape(4, 2 * 3) # 4 is the dimensions of a that do not require tensordot
        >>> reshaped_b = permuted_b.reshape(2 * 3, 4) # 4 is the dimensions of a that do not require tensordot
        >>> matmul_result = flow.matmul(reshaped_a, reshaped_b)
        >>> result = matmul_result.reshape(4, 4) # 4, 4 are the concatentation of dimensions that do not require tensordot of a and b
        >>> flow.all(result == flow.tensordot(a, b, dims))
        tensor(True, dtype=oneflow.bool)

    """,
)
