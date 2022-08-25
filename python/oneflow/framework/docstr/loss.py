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
    cross_entropy(input, target, weight=None, ignore_index=-100, reduction="mean")

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

add_docstr(
    oneflow._C.l1_loss,
    r"""
    l1_loss(input, target, reduction="mean") -> Tensor

    This operator computes the L1 loss between each element in input and target.

    see :class:`~oneflow.nn.L1Loss` for details.

    Args:
        input (Tensor): The input Tensor.
        target (Tensor): The target Tensor.
        reduction (string, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".
    
    Examples::

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, 4, requires_grad=True)
        >>> target = flow.rand(3, 4, requires_grad=False)
        >>> loss = F.l1_loss(input, target)
        >>> loss.backward()

    """,
)

add_docstr(
    oneflow._C.mse_loss,
    r"""
    mse_loss(input, target, reduction="mean") -> Tensor

    This operator computes the mean squared error (squared L2 norm) 
    loss between each element in input and target.

    see :class:`~oneflow.nn.MSELoss` for details.

    Args:
        input (Tensor): The input Tensor.
        target (Tensor): The target Tensor.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Examples::

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, 4, requires_grad=True)
        >>> target = flow.rand(3, 4, requires_grad=False)
        >>> loss = F.mse_loss(input, target)
        >>> loss.backward()

    """,
)

add_docstr(
    oneflow._C.smooth_l1_loss,
    """
    smooth_l1_loss(input: Tensor, target: Tensor, size_average: bool=True, reduce: bool=True, reduction: str='mean', beta: float=1.0) -> Tensor

    Function that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    See :class:`~oneflow.nn.SmoothL1Loss` for details.
    """,
)

add_docstr(
    oneflow._C.binary_cross_entropy_loss,
    r"""
    binary_cross_entropy(input, target, weight=None, reduction='mean')

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.binary_cross_entropy.html.
    
    Function that measures the Binary Cross Entropy between the target and input probabilities.

    See :class:`~oneflow.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape as probabilities.
        target: Tensor of the same shape as input with values between 0 and 1.
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, 2, requires_grad=True)
        >>> target = flow.rand(3, 2, requires_grad=False)
        >>> loss = F.binary_cross_entropy(flow.sigmoid(input), target)
        >>> loss.backward()
    """,
)

add_docstr(
    oneflow._C.binary_cross_entropy_with_logits_loss,
    r"""
    binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None)

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.binary_cross_entropy_with_logits.html.

    Function that measures Binary Cross Entropy between target and input logits.

    See :class:`~oneflow.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.randn(3, requires_grad=True)
        >>> target = flow.randn(3)
        >>> target[target >= 0] = 1
        >>> target[target < 0] = 0
        >>> loss = F.binary_cross_entropy_with_logits(input, target)
        >>> loss.backward()
    """,
)

add_docstr(
    oneflow._C.ctc_loss,
    r"""
    ctc_loss(log_probs, target, input_lenghts, target_lengths, blank=0, zero_infinity=False, reduction="mean")

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.ctc_loss.html

    See :class:`~oneflow.nn.CTCLoss` for details.

    Args:
        log_probs (Tensor): :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`oneflow.nn.functional.log_softmax`).
        targets (Tensor): :math:`(N, S)` or `(sum(target_lengths))`.
            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths (Tensor): :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths (Tensor): :math:`(N)`.
            Lengths of the targets
        blank (int, optional): blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Shape:
        - Log_probs: Tensor of size :math:`(T, N, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> log_probs = flow.tensor(
        ...    [
        ...        [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
        ...        [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
        ...        [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
        ...        [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
        ...        [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
        ...    ], dtype=flow.float32)
        >>> targets = flow.tensor([[1, 2, 2], [1, 2, 2]], dtype=flow.int32)
        >>> input_lengths = flow.tensor([5, 5], dtype=flow.int32)
        >>> target_lengths = flow.tensor([3, 3], dtype=flow.int32)
        >>> out = flow.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(1.1376, dtype=oneflow.float32)
        >>> out = flow.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="sum")
        >>> out
        tensor(6.8257, dtype=oneflow.float32)

    """
)