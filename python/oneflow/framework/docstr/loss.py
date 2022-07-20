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
    oneflow._C.triplet_margin_loss,
    r"""    
    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.
    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.triplet_margin_loss.html.
    
    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses <http://www.bmva.org/bmvc/2016/papers/paper119/index.html>`__ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: :math:`1`.
        p (float, optional): The norm degree for pairwise distance. Default: :math:`2.0`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
          otherwise.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> triplet_loss = flow.nn.TripletMarginLoss(margin=1.0, p=2)
        >>> anchor = np.array([[1, -1, 1],[-1, 1, -1], [1, 1, 1]])
        >>> positive = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> negative = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        >>> output = triplet_loss(flow.Tensor(anchor), flow.Tensor(positive), flow.Tensor(negative))
        >>> output
        tensor(6.2971, dtype=oneflow.float32)
    
    """,
)

add_docstr(
    oneflow._C.cross_entropy,
    r"""
    See :class:`~oneflow.nn.CrossEntropyLoss` for details.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.cross_entropy.html.


    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to contain unnormalized scores
            (often referred to as logits).
        target (Tensor) : If containing class indices, shape :math:`(N)` where each value is
            :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
            :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
            same shape as the input.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
            Default: -100
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, 5, requires_grad=True)
        >>> target = flow.ones(3, dtype=flow.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()


    """,
)
