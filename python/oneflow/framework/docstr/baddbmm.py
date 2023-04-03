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
    oneflow.baddbmm,
    r"""
    baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.baddbmm.html.

    Performs a batch matrix-matrix product of matrices in :attr:`batch1` and :attr:`batch2`.
    :attr:`input` is added to the final result.

    :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
    number of matrices.

    If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
    :math:`(b \times m \times p)` tensor, then :attr:`input` must be
    broadcastable with a
    :math:`(b \times n \times p)` tensor and :attr:`out` will be a
    :math:`(b \times n \times p)` tensor.

    .. math::
        \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)

    If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in it will not be propagated.

    For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
    :attr:`alpha` must be real numbers, otherwise they should be integers.

    Args:
    input (Tensor): the tensor to be added
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied

    Keyword args:
        beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
        alpha (Number, optional): multiplier for :math:`\text{{batch1}} \mathbin{{@}} \text{{batch2}}` (:math:`\alpha`)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.randn(10, 3, 5)
        >>> batch1 = flow.randn(10, 3, 4)
        >>> batch2 = flow.randn(10, 4, 5)
        >>> of_out = flow.baddbmm(input, batch1, batch2)
        >>> of_out.shape
        oneflow.Size([10, 3, 5])
    """,
)
