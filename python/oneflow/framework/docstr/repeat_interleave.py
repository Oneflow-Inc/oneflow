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
    oneflow.repeat_interleave,
    """
    repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave.html

    Repeat elements of a tensor.

    .. warning::

        This is different from :meth:`oneflow.Tensor.repeat` but similar to ``numpy.repeat``.

    Args:
        input (oneflow.Tensor): the input Tensor.
        repeats (Tensor or int): The number of repetitions for each element.
            repeats is broadcasted to fit the shape of the given axis.
        dim (int, optional): The dimension along which to repeat values.
            By default, use the flattened input array, and return a flat output
            array.

    Keyword args:
        output_size (int, optional): Total output size for the given axis
            ( e.g. sum of repeats). If given, it will avoid stream syncronization
            needed to calculate output shape of the tensor.

    Returns:
        oneflow.Tensor: Repeated tensor which has the same shape as input, except along the given axis.
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        tensor([1, 2, 3], dtype=oneflow.int64)
        >>> y = flow.tensor([[1, 2], [3, 4]])
        >>> flow.repeat_interleave(y, 2)
        tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, 3, dim=1)
        tensor([[1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4]], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, flow.tensor([1, 2]), dim=0)
        tensor([[1, 2],
                [3, 4],
                [3, 4]], dtype=oneflow.int64)
        >>> flow.repeat_interleave(y, flow.tensor([1, 2]), dim=0, output_size=3)
        tensor([[1, 2],
                [3, 4],
                [3, 4]], dtype=oneflow.int64)
    """,
)

