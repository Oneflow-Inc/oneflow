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
    oneflow.lerp,
    """
    lerp(start, end, weight) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.lerp.html.

    Does a linear interpolation of two tensors `start` and `end` based on a scalar or tensor `weight` and returns the result.

    The shapes of start` and `end` must be broadcastable. If `weight` is a tensor, then the shapes of `weight`, `start`, and `end` must be broadcastable.

    .. math::
        out_{i} = start_{i} + weight_{i} * (end_{i} - start_{i})

    Args:
        start (oneflow.Tensor): the tensor with the starting points.
        end (oneflow.Tensor): the tensor with the ending points.
        weight (float or oneflow.Tensor): the weight for the interpolation formula.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> start = flow.arange(1., 5.)
        >>> end = flow.empty(4).fill_(10)
        >>> flow.lerp(start, end, 0.5)
        tensor([5.5000, 6.0000, 6.5000, 7.0000], dtype=oneflow.float32)
        >>> flow.lerp(start, end, flow.full_like(start, 0.5))
        tensor([5.5000, 6.0000, 6.5000, 7.0000], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.lerp_,
    """
    In-place version of :func:`oneflow.lerp`
    """,
)
