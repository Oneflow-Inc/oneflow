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
    oneflow.split,
    """Splits the tensor into chunks.

    If `split_size_or_sections` is an integer type, then x will be split into equally sized chunks (if possible).
    Last chunk will be smaller if the tensor size along the given dimension `dim` is not divisible by split_size.

    If `split_size_or_sections` is a list, then x will be split into `len(split_size_or_sections)` chunks
    with sizes in `dim` according to `split_size_or_sections`.

    Args:
        x: tensor to split.
        split_size_or_sections: size of a single chunk or list of sizes for each chunk.
        dim: dimension along which to split the tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.arange(10).view(5, 2)
        >>> flow.split(a, 2)
        (tensor([[0, 1],
                [2, 3]], dtype=oneflow.int64), tensor([[4, 5],
                [6, 7]], dtype=oneflow.int64), tensor([[8, 9]], dtype=oneflow.int64))
        >>> flow.split(a, [1, 4])
        (tensor([[0, 1]], dtype=oneflow.int64), tensor([[2, 3],
                [4, 5],
                [6, 7],
                [8, 9]], dtype=oneflow.int64))
    """,
)
