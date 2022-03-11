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
    oneflow.split_with_sizes,
    """
    split_with_sizes(x, split_sizes, dim) -> TensorTuple

    This function is equivalent to torch’s split_with_sizes function.

    `x` will be split into `len(split_sizes)` chunks
    with sizes in `dim` according to `split_sizes`.

    Args:
        x: tensor to split.
        split_sizes: list of sizes for each chunk.
        dim: dimension along which to split the tensor.

    For example:

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> a = flow.arange(10).reshape(5,2)
        >>> flow.split_with_sizes(a, [1,4])
        (tensor([[0, 1]], dtype=oneflow.int64), tensor([[2, 3],
                [4, 5],
                [6, 7],
                [8, 9]], dtype=oneflow.int64))

    """,
)
