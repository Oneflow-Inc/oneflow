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
from typing import Optional

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.modules.module import Module
from oneflow.nn.modules.constant import _ConstantBase


class _Loss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        assert reduction in ["none", "mean", "sum"]
        self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self, weight: Optional[Tensor] = None, reduction: str = "mean"
    ) -> None:
        super(_WeightedLoss, self).__init__(reduction=reduction)
        self.register_buffer("weight", weight)


class L1Loss(_Loss):
    """This operator computes the L1 Loss between each element in `input` and `target`.

    The equation is:

    if reduction = "none":

    .. math::

        output = |Target - Input|

    if reduction = "mean":

    .. math::

        output = \\frac{1}{n}\\sum_{i=1}^n|Target_i - Input_i|

    if reduction = "sum":

    .. math::

        output = \\sum_{i=1}^n|Target_i - Input_i|

    Args:
        input (oneflow.Tensor): the input Tensor.
        target (oneflow.Tensor): The target Tensor.
        reduction (str): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor([[1, 1, 1], [2, 2, 2], [7, 7, 7]], dtype = flow.float32)
        >>> target = flow.tensor([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype = flow.float32)
        >>> m = flow.nn.L1Loss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[3., 3., 3.],
                [2., 2., 2.],
                [3., 3., 3.]], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="mean")
        >>> out = m_mean(input, target)
        >>> out
        tensor(2.6667, dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="sum")
        >>> out = m_mean(input, target)
        >>> out
        tensor(24., dtype=oneflow.float32)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(L1Loss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.l1_loss(input, target, self.reduction)


class CrossEntropyLoss(_WeightedLoss):
    r"""
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.CrossEntropyLoss.html.

    This criterion combines :class:`~flow.nn.LogSoftmax` and :class:`~flow.nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument `weight` should be a 1D Tensor assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.
    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    The target that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
               \text{if reduction} = \text{'mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{'sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{'mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{'sum'.}
            \end{cases}


    Args:
        weight (oneflow.Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored and does not
            contribute to the input gradient. When ``reduction`` is ``mean``, the loss is averaged
            over non-ignored targets. Note that ``ignore_index`` is only applicable when the target
            contains class indices.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``
        label_smoothing (float, optinoal): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing.
            The targets become a mixture of the original ground truth and a uniform
            distribution as described in `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`_.
            Default: :math:`0.0`.

    Shape:
        - Input: Shape ::math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
        - Output: If reduction is 'none', same shape as the target. Otherwise, scalar.

        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}


    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(
        ...    [[-0.1664078, -1.7256707, -0.14690138],
        ...        [-0.21474946, 0.53737473, 0.99684894],
        ...        [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
        >>> out
        tensor([0.8020, 1.1167, 0.3583], dtype=oneflow.float32)
        >>> out_sum = flow.nn.CrossEntropyLoss(reduction="sum")(input, target)
        >>> out_sum
        tensor(2.2769, dtype=oneflow.float32)
        >>> out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
        >>> out_mean
        tensor(0.7590, dtype=oneflow.float32)
        >>> out_ignore_0 = flow.nn.CrossEntropyLoss(reduction="none", ignore_index=0)(input, target)
        >>> out_ignore_0
        tensor([0.0000, 1.1167, 0.3583], dtype=oneflow.float32)
        >>> out_label_smoothing = flow.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.5)(input, target)
        >>> out_label_smoothing
        tensor([1.0586, 1.1654, 0.8864], dtype=oneflow.float32)
        >>> probs = flow.tensor([[ 0.99495536,  0.28255007, -0.2775054 ],
        ...    [ 0.42397153,  0.01075112,  0.56527734],
        ...    [ 0.72356546, -0.1304398 ,  0.4068744 ]], dtype=flow.float32)
        >>> out = flow.nn.CrossEntropyLoss()(input, probs)
        >>> out
        tensor(1.3305, dtype=oneflow.float32)

    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if self.label_smoothing < 0.0 or self.label_smoothing > 1.0:
            raise ValueError(
                "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing
            )

    def forward(self, input, target):
        return flow._C.cross_entropy(
            input,
            target,
            self.weight,
            self.ignore_index,
            self.reduction,
            self.label_smoothing,
        )


class BCELoss(_WeightedLoss):
    """This operator computes the binary cross entropy loss.

    The equation is:

    if reduction = "none":

    .. math::

        out = -(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    if reduction = "mean":

    .. math::

        out = -\\frac{1}{n}\\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    if reduction = "sum":

    .. math::

        out = -\\sum_{i=1}^n(Target_i*log(Input_i) + (1-Target_i)*log(1-Input_i))

    Args:
        weight (oneflow.Tensor, optional): The manual rescaling weight to the loss. Default to None, whose corresponding weight value is 1.
        reduction (str, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Attention:
        The input value must be in the range of (0, 1). Or the loss function may return `nan` value.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32))
        >>> target = flow.Tensor(np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32))
        >>> weight = flow.Tensor(np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32))
        >>> activation = flow.nn.Sigmoid()
        >>> sigmoid_input = activation(input)
        >>> m = flow.nn.BCELoss(weight, reduction="none")
        >>> out = m(sigmoid_input, target)
        >>> out
        tensor([[2.9266, 1.1963, 1.1087],
                [0.8064, 2.0750, 4.2539]], dtype=oneflow.float32)
        >>> m_sum = flow.nn.BCELoss(weight, reduction="sum")
        >>> out = m_sum(sigmoid_input, target)
        >>> out
        tensor(12.3668, dtype=oneflow.float32)
        >>> m_mean = flow.nn.BCELoss(weight, reduction="mean")
        >>> out = m_mean(sigmoid_input, target)
        >>> out
        tensor(2.0611, dtype=oneflow.float32)
        >>> m_none = flow.nn.BCELoss()
        >>> out = m_none(sigmoid_input, target)
        >>> out
        tensor(1.0306, dtype=oneflow.float32)

    """

    def __init__(
        self, weight: Optional[Tensor] = None, reduction: str = "mean"
    ) -> None:
        super(BCELoss, self).__init__(weight, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.binary_cross_entropy_loss(
            input, target, self.weight, self.reduction
        )


class NLLLoss(_WeightedLoss):
    """ The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \\geq 1` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`;

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = - w_{y_n} x_{n,y_n}, \\quad
        w_{c} = \\mathbb{1},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \\ell(x, y) = \\begin{cases}
            \\sum_{n=1}^N \\frac{1}{N} l_n, &
            \\text{if reduction} = \\text{`mean';}\\\\
            \\sum_{n=1}^N l_n,  &
            \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(
        ... [[-0.1664078, -1.7256707, -0.14690138],
        ... [-0.21474946, 0.53737473, 0.99684894],
        ... [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> m = flow.nn.NLLLoss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([ 0.1664, -0.5374, -0.7645], dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor(-1.1355, dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor(-0.3785, dtype=oneflow.float32)

    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super(NLLLoss, self).__init__(weight, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.nll_loss(
            input, target, self.weight, self.ignore_index, self.reduction
        )


class KLDivLoss(_Loss):
    """
    The Kullback-Leibler divergence loss measure

    `Kullback-Leibler divergence`_ is a useful distance measure for continuous
    distributions and is often useful when performing direct regression over
    the space of (discretely sampled) continuous output distributions.

    As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
    *log-probabilities* and is not restricted to a 2D Tensor.
    The targets are interpreted as *probabilities* by default, but could be considered
    as *log-probabilities* with :attr:`log_target` set to ``True``.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        l(x,y) = L = \\{ l_1,\\dots,l_N \\}, \\quad
        l_n = y_n \\cdot \\left( \\log y_n - x_n \\right)

    where the index :math:`N` spans all dimensions of ``input`` and :math:`L` has the same
    shape as ``input``. If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        \\ell(x, y) = \\begin{cases}
            \\operatorname{mean}(L), & \\text{if reduction} = \\text{`mean';} \\\\
            \\operatorname{sum}(L),  & \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    In default :attr:`reduction` mode ``'mean'``, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. ``'batchmean'`` mode gives the correct KL divergence where losses
    are averaged over batch dimension only. ``'mean'`` mode's behavior will be changed to the same as
    ``'batchmean'`` in the next major release.

    .. _`kullback-leibler divergence`: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.KLDivLoss.html.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied.
            ``'batchmean'``: the sum of the output will be divided by batchsize.
            ``'sum'``: the output will be summed.
            ``'mean'``: the output will be divided by the number of elements in the output.
            Default: ``'mean'``
        log_target (bool, optional): Specifies whether `target` is passed in the log space.
            Default: ``False``

    .. note::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as ``'batchmean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, *)`,
          the same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor([-0.9021705, 0.08798598, 1.04686249], dtype=flow.float32)
        >>> target = flow.tensor([1.22386942, -0.89729659, 0.01615712], dtype=flow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="none", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor([ 1.3514,  0.0000, -0.0836], dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="mean", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor(0.4226, dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="sum", log_target=True)
        >>> out = m(input, target)
        >>> out
        tensor(5.7801, dtype=oneflow.float32)

    """

    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        if reduction == "batchmean":
            super(KLDivLoss, self).__init__("sum")
            self.reduction = "batchmean"
        else:
            super(KLDivLoss, self).__init__(reduction)

        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.kl_div_loss(input, target, self.log_target, self.reduction)


class MSELoss(_Loss):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = \\left( x_n - y_n \\right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\operatorname{mean}(L), &  \\text{if reduction} = \\text{`mean';}\\\\
            \\operatorname{sum}(L),  &  \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.MSELoss.html.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(
        ... [[-0.02557137, 0.03101675, 1.37493674],
        ... [0.25599439, -1.08372561, -0.21006816]], dtype=flow.float32)
        >>> target = flow.tensor(
        ... [[-1.53105064, -0.68137555, 0.5931354],
        ... [-0.49158347, 0.93673637, 0.1324141]], dtype=flow.float32)
        >>> m = flow.nn.MSELoss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[2.2665, 0.5075, 0.6112],
                [0.5589, 4.0823, 0.1173]], dtype=oneflow.float32)
        >>> m = flow.nn.MSELoss(reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor(1.3573, dtype=oneflow.float32)
        >>> m = flow.nn.MSELoss(reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor(8.1436, dtype=oneflow.float32)

    """

    def __init__(self, reduction: str = "mean") -> None:
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.mse_loss(input, target, self.reduction)


class MarginRankingLoss(_Loss):
    """Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        \\text{loss}(x1, x2, y) = \\max(0, -y * (x1 - x2) + \\text{margin})

    Args:
        margin (float, optional): Has a default value of :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - `x1` : :math:`(N, D)` where `N` is the batch size and `D` is the size of a sample.
        - `x2` : :math:`(N, D)` where `N` is the batch size and `D` is the size of a sample.
        - Target: :math:`(N)`
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=flow.float32)
        >>> x2 = flow.tensor(np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]), dtype=flow.float32)
        >>> target = flow.tensor(np.array([[1, -1, 1],[-1, 1, -1], [1, 1, 1]]), dtype=flow.float32)
        >>> m = flow.nn.MarginRankingLoss(margin =1.0, reduction="none")
        >>> out = m(x1, x2, target)
        >>> out
        tensor([[2., 1., 0.],
                [3., 0., 5.],
                [0., 0., 0.]], dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 0.3, reduction="sum")
        >>> out = m(x1, x2, target)
        >>> out
        tensor(8.2000, dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 10, reduction="mean")
        >>> out = m(x1, x2, target)
        >>> out
        tensor(8.3333, dtype=oneflow.float32)


    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super(MarginRankingLoss, self).__init__(reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return flow._C.margin_ranking_loss(
            input1, input2, target, self.margin, self.reduction
        )


class CTCLoss(_Loss):
    """The Connectionist Temporal Classification loss.
    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.CTCLoss.html.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\\leq` the input length.

    Args:
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
          where :math:`T = \\text{input length}`,
          :math:`N = \\text{batch size}`, and
          :math:`C = \\text{number of classes (including blank)}`.
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\\operatorname{sum}(\\text{target_lengths}))`,
          where :math:`N = \\text{batch size}` and
          :math:`S = \\text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\\operatorname{sum}(\\text{target_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \\text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \\text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\\leq S`
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
        >>> loss_mean = flow.nn.CTCLoss()
        >>> out = loss_mean(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(1.1376, dtype=oneflow.float32)
        >>> loss_sum = flow.nn.CTCLoss(blank=0, reduction="sum")
        >>> out = loss_sum(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor(6.8257, dtype=oneflow.float32)

    """

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False
    ) -> None:
        super(CTCLoss, self).__init__(reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        max_target_length = 0
        if targets.ndim == 1:
            max_target_length = target_lengths.max().item()
        elif targets.ndim == 2:
            max_target_length = targets.shape[1]
        return flow._C.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            max_target_length,
            self.blank,
            self.zero_infinity,
            self.reduction,
        )


class BCEWithLogitsLoss(_WeightedLoss):
    """This operator combines the `Sigmoid` and `BCELoss` together. For numerical stability,
    we apply some math tricks instead of using `Sigmoid` layer with `BCELoss`.

    The equation is:

    if :attr:`reduction` = ``"none"``:

    .. math::

        out = -weight*[Pos\\_weight*y*log\\sigma({x}) + (1-y)*log(1-\\sigma(x))]

    if :attr:`reduction` = ``"mean"``:

    .. math::

        out = -\\frac{weight}{n}\\sum_{i=1}^n[Pos\\_weight*y*log\\sigma({x}) + (1-y)*log(1-\\sigma(x))]

    if :attr:`reduction` = ``"sum"``:

    .. math::

        out = -weight*\\sum_{i=1}^n[Pos\\_weight*y*log\\sigma({x}) + (1-y)*log(1-\\sigma(x))]

    Args:
        weight (Tensor, optional): The manual rescaling weight to the loss. Default: ``None``
        size_average (bool, optional): Deprecated (see :attr:`reduction`). Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). Default: ``True``
        reduction (str, optional): The reduce type, it can be one of ``"none"``, ``"mean"``, ``"sum"``.
            ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be divided
            by the number of elements in the output, ``'sum'``: the output will be summed. Default: ``"mean"``
        pos_weight (Tensor, optional): The manual rescaling weight to the positive examples.
            Default: ``None``

    Shape:
        - Input: :math:`(N,*)` where `*` means, any number of additional dimensions
        - Target: :math:`(N,*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``"none"``, then :math:`(N,*)`, same shape as input.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1.2, 0.2, -0.3], [0.7, 0.6, -2], [0.7, 0.6, -2]], dtype=flow.float32)
        >>> target = flow.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=flow.float32)
        >>> weight = flow.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=flow.float32)
        >>> pos_weight = flow.tensor([1.2, 1.3, 1.4], dtype=flow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[2.9266, 1.5552, 1.1087],
                [0.9676, 2.0750, 5.9554],
                [0.9676, 2.0750, 5.9554]], dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor(2.6207, dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor(23.5865, dtype=oneflow.float32)


    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super(BCEWithLogitsLoss, self).__init__(weight, reduction)
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.binary_cross_entropy_with_logits_loss(
            input, target, self.weight, self.pos_weight, self.reduction
        )


class SmoothL1Loss(_Loss):
    """Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.
    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.SmoothL1Loss.html.

    It is less sensitive to outliers than :class:`torch.nn.MSELoss` and in some cases
    prevents exploding gradients (e.g. see the paper `Fast R-CNN <https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf>`__ by Ross Girshick)..

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1, ..., l_N\\}^T

    with

    .. math::
        l_n = \\begin{cases}
        0.5 (x_n - y_n)^2 / beta, & \\text{if } |x_n - y_n| < beta \\\\
        |x_n - y_n| - 0.5 * beta, & \\text{otherwise }
        \\end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\operatorname{mean}(L), &  \\text{if reduction} = \\text{`mean';}\\\\
            \\operatorname{sum}(L),  &  \\text{if reduction} = \\text{`sum'.}
        \\end{cases}

    .. note::
        Smooth L1 loss can be seen as exactly :class:`L1Loss`, but with the :math:`|x - y| < beta`
        portion replaced with a quadratic function such that its slope is 1 at :math:`|x - y| = beta`.
        The quadratic segment smooths the L1 loss near :math:`|x - y| = 0`.

    .. note::
        Smooth L1 loss is closely related to :class:`HuberLoss`, being
        equivalent to :math:`huber(x, y) / beta` (note that Smooth L1's beta hyper-parameter is
        also known as delta for Huber). This leads to the following differences:

        * As beta -> 0, Smooth L1 loss converges to :class:`L1Loss`, while :class:`HuberLoss`
          converges to a constant 0 loss.
        * As beta -> :math:`+\\infty`, Smooth L1 loss converges to a constant 0 loss, while
          :class:`HuberLoss` converges to :class:`MSELoss`.
        * For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1.
          For :class:`HuberLoss`, the slope of the L1 segment is beta.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional dimensions
        - Target: :math:`(N, *)`; same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`; same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.tensor(np.array([0.1, 0.4, 0.3, 0.5, 0.9]).astype(np.float32), dtype=flow.float32)
        >>> y = flow.tensor(np.array([0.3, 0.9, 2.5, 0.4, 0.3]).astype(np.float32), dtype=flow.float32)
        >>> m = flow.nn.SmoothL1Loss(reduction="none")
        >>> out = m(x, y)
        >>> out
        tensor([0.0200, 0.1250, 1.7000, 0.0050, 0.1800], dtype=oneflow.float32)

        >>> m = flow.nn.SmoothL1Loss(reduction="mean")
        >>> out = m(x, y)
        >>> out
        tensor(0.4060, dtype=oneflow.float32)

        >>> m = flow.nn.SmoothL1Loss(reduction="sum")
        >>> out = m(x, y)
        >>> out
        tensor(2.0300, dtype=oneflow.float32)
    """

    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super(SmoothL1Loss, self).__init__(reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return flow._C.smooth_l1_loss(input, target, self.beta, self.reduction)


class CombinedMarginLoss(Module):
    r"""The operation implements "margin_softmax" in InsightFace:
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_mxnet/train.py
    The implementation of margin_softmax in InsightFace is composed of multiple operators.
    We fuse them for speed up.

    Applies the function:

    .. math::

        {\rm CombinedMarginLoss}(x_i, label) =
        \left\{\begin{matrix} \cos(m_1\cdot\arccos x_i+m_2) - m_3 & {\rm if} \ i == label \\
        x_i & {\rm otherwise} \end{matrix}\right.


    Args:
        x (oneflow.Tensor): A Tensor
        label (oneflow.Tensor): label with integer data type
        m1 (float): loss m1 parameter
        m2 (float): loss m2 parameter
        m3 (float): loss m3 parameter

    .. note::

        Here are some special cases:

        - when :math:`m_1=1, m_2\neq 0, m_3=0`, CombineMarginLoss has the same parameter as `ArcFace <https://arxiv.org/abs/1801.07698>`__ .

        - when :math:`m_1=1, m_2=0, m_3\neq 0`, CombineMarginLoss has the same parameter as `CosFace (a.k.a AM-Softmax) <https://arxiv.org/abs/1801.09414>`__ .

        - when :math:`m_1\gt 1, m_2=m_3=0`, CombineMarginLoss has the same parameter as `A-Softmax <https://arxiv.org/abs/1704.08063>`__.

    Returns:
        oneflow.Tensor: A Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> np_x = np.array([[-0.7027179, 0.0230609], [-0.02721931, -0.16056311], [-0.4565852, -0.64471215]])
        >>> np_label = np.array([0, 1, 1])
        >>> x = flow.tensor(np_x, dtype=flow.float32)
        >>> label = flow.tensor(np_label, dtype=flow.int32)
        >>> loss_func = flow.nn.CombinedMarginLoss(0.3, 0.5, 0.4)
        >>> out = loss_func(x, label)
        >>> out
        tensor([[-0.0423,  0.0231],
                [-0.0272,  0.1237],
                [-0.4566, -0.0204]], dtype=oneflow.float32)

    """

    def __init__(self, m1: float = 1.0, m2: float = 0.0, m3: float = 0.0) -> None:
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        return flow._C.combined_margin_loss(
            x, label, m1=self.m1, m2=self.m2, m3=self.m3
        )


class TripletMarginLoss(Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

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

    """

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        triplet_loss = flow._C.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )
        return triplet_loss


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
