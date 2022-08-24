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
    oneflow.unbind,
    """
    Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.
    
    This function is equivalent to PyTorch's unbind function.

    Args:
        x(Tensor): the tensor to unbind
        dim(int): dimension to remove

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> x = flow.tensor(range(12)).reshape([3,4])
        >>> flow.unbind(x)
        (tensor([0, 1, 2, 3], dtype=oneflow.int64), tensor([4, 5, 6, 7], dtype=oneflow.int64), tensor([ 8,  9, 10, 11], dtype=oneflow.int64))
        >>> flow.unbind(x, 1)
        (tensor([0, 4, 8], dtype=oneflow.int64), tensor([1, 5, 9], dtype=oneflow.int64), tensor([ 2,  6, 10], dtype=oneflow.int64), tensor([ 3,  7, 11], dtype=oneflow.int64))

    """,
)
