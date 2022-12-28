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
    oneflow.searchsorted,
    """
    searchsorted() -> oneflow.Tensor

    Find the indices from the innermost dimension of sorted_sequence such that, if the corresponding values
    in values were inserted before the indices, the order of the corresponding innermost dimension within
    sorted_sequence would be preserved. Return a new tensor with the same size as values. If right is False
    (default), then the left boundary of sorted_sequence is closed. More formally, the returned index
    satisfies the following rules:

    =================  =========  ==========================================================================
    sorted_sequence     right      returned index satisfies
    =================  =========  ==========================================================================
    1-D                 False      sorted_sequence[i-1] < values[m][n]...[l][x] <= sorted_sequence[i]
    1-D                 True       sorted_sequence[i-1] <= values[m][n]...[l][x] < sorted_sequence[i]
    N-D                 False      sorted_sequence[m][n]...[l][i-1] < values[m][n]...[l][x] 
                                                    <= sorted_sequence[m][n]...[l][i]
    N-D                 True       sorted_sequence[m][n]...[l][i-1] <= values[m][n]...[l][x] 
                                                    sorted_sequence[m][n]...[l][i]
    =================  =========  ==========================================================================

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.searchsorted.html

    Args:
        sorted_sequence (Tensor): N-D or 1-D tensor, containing monotonically increasing sequence on the
                                innermost dimension.
        values (Tensor or Scalar): N-D tensor or a Scalar containing the search value(s).
        out_int32 (bool optional): indicate the output data type. torch.int32 if True, torch.int64 otherwise.
                                Default value is False, i.e. default output data type is torch.int64.
        right (bool optional): if False, return the first suitable location that is found. If True, return the
                                last such index. If no suitable index found, return 0 for non-numerical value
                                (eg. nan, inf) or the size of innermost dimension within sorted_sequence (one
                                pass the last index of the innermost dimension). In other words, if False, gets
                                the lower bound index for each value in values on the corresponding innermost
                                dimension of the sorted_sequence. If True, gets the upper bound index instead.
                                Default value is False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> sorted_sequence = flow.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        >>> sorted_sequence
        tensor([[ 1,  3,  5,  7,  9],
                [ 2,  4,  6,  8, 10]], dtype=oneflow.int64)
        >>> values = flow.tensor([[3, 6, 9], [3, 6, 9]])
        >>> values
        tensor([[3, 6, 9],
                [3, 6, 9]], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence, values)
        tensor([[1, 3, 4],
                [1, 2, 4]], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence, values, right=True)
        tensor([[2, 3, 5],
                [1, 3, 4]], dtype=oneflow.int64)
        >>> sorted_sequence_1d = flow.tensor([1, 3, 5, 7, 9])
        >>> sorted_sequence_1d
        tensor([1, 3, 5, 7, 9], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence_1d, values)
        tensor([[1, 3, 4],
                [1, 3, 4]], dtype=oneflow.int64)

    """,
)
