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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import Tensor
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.constant import _ConstantBase


class L1Loss(Module):
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
        input (oneflow.compatible.single_client.experimental.Tensor): The input Tensor.
        target (oneflow.compatible.single_client.experimental.Tensor): The target Tensor.
        reduction (str): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Returns:
        oneflow.compatible.single_client.experimental.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor([[1, 1, 1], [2, 2, 2], [7, 7, 7]], dtype = flow.float32)
        >>> target = flow.Tensor([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype = flow.float32)
        >>> m = flow.nn.L1Loss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[3., 3., 3.],
                [2., 2., 2.],
                [3., 3., 3.]], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="mean")
        >>> out = m_mean(input, target)
        >>> out
        tensor([2.6667], dtype=oneflow.float32)
        >>> m_mean = flow.nn.L1Loss(reduction="sum")
        >>> out = m_mean(input, target)
        >>> out
        tensor([24.], dtype=oneflow.float32)
    """

    def __init__(self, reduction: str = "mean", reduce=True) -> None:
        super().__init__()
        if reduce is not None and (not reduce):
            raise ValueError("Argument reduce is not supported yet")
        assert reduction in [
            "none",
            "mean",
            "sum",
            None,
        ], "only 'sum', 'mean' and 'none' supported by now"
        self.reduction = reduction

    def forward(self, input, target):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"
        l1_value = flow.experimental.abs(flow.experimental.sub(input, target))
        if self.reduction == "mean":
            return flow.experimental.mean(l1_value)
        elif self.reduction == "sum":
            return flow.experimental.sum(l1_value)
        else:
            return l1_value


class CrossEntropyLoss(Module):
    """This criterion combines :class:`~flow.nn.LogSoftmax` and :class:`~flow.nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \\geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`;

    The loss can be described as:

    .. math::
        \\text{loss}(x, class) = -\\log\\left(\\frac{\\exp(x[class])}{\\sum_j \\exp(x[j])}\\right)
                       = -x[class] + \\log\\left(\\sum_j \\exp(x[j])\\right)

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(
        ...    [[-0.1664078, -1.7256707, -0.14690138],
        ...        [-0.21474946, 0.53737473, 0.99684894],
        ...        [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
        >>> out
        tensor([0.802 , 1.1167, 0.3583], dtype=oneflow.float32)
        >>> out_sum = flow.nn.CrossEntropyLoss(reduction="sum")(input, target)
        >>> out_sum
        tensor([2.2769], dtype=oneflow.float32)
        >>> out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
        >>> out_mean
        tensor([0.759], dtype=oneflow.float32)

    """

    def __init__(
        self,
        weight=None,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super().__init__()
        if weight is not None:
            raise ValueError("Argument weight is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        assert len(input.shape) <= 4
        assert len(target.shape) == len(input.shape) - 1
        input_shape_len = len(input.shape)
        if input_shape_len == 3:
            (b, c, h) = (input.shape[0], input.shape[1], input.shape[2])
            input = flow.F.transpose(input, perm=(0, 2, 1))
            input = input.reshape(shape=[-1, input.shape[2]])
            target = target.flatten()
        elif input_shape_len == 4:
            (b, c, h, w) = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
                input.shape[3],
            )
            input = flow.F.transpose(input, perm=(0, 2, 3, 1))
            input = input.reshape(shape=[-1, input.shape[3]])
            target = target.flatten()
        elif input_shape_len >= 5:
            raise NotImplemented
        out = flow.F.sparse_softmax_cross_entropy(
            input, target, depth=input.shape[len(input.shape) - 1]
        )
        if self.ignore_index is not None:
            zeros = flow.experimental.zeros(
                size=out.shape, dtype=out.dtype, device=out.device
            )
            condition = flow.experimental.eq(target, self.ignore_index)
            ones = flow.experimental.ones(
                size=condition.shape, dtype=condition.dtype, device=condition.device
            )
            condition = ones.sub(condition).reshape(tuple(out.shape))
            out = flow.experimental.where(condition, out, zeros)
            if self.reduction == "mean":
                reduce_sum = out.sum()
                reduce_count = condition.argwhere().shape[0]
                out = flow.experimental.mul(reduce_sum, 1.0 / reduce_count)
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            if input_shape_len == 4:
                out = out.reshape((b, h, w))
            return out


class BCELoss(Module):
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
        weight (oneflow.compatible.single_client.experimental.Tensor, optional): The manual rescaling weight to the loss. Default to None, whose corresponding weight value is 1.
        reduction (str, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".

    Attention:
        The input value must be in the range of (0, 1). Or the loss function may return `nan` value.

    Returns:
        oneflow.compatible.single_client.experimental.Tensor: The result Tensor.

    For example:

    .. code-block:: python
    
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([[1.2, 0.2, -0.3], [0.7, 0.6, -2]]).astype(np.float32))
        >>> target = flow.Tensor(np.array([[0, 1, 0], [1, 0, 1]]).astype(np.float32))
        >>> weight = flow.Tensor(np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32))
        >>> activation = flow.nn.Sigmoid()
        >>> sigmoid_input = activation(input)
        >>> m = flow.nn.BCELoss(weight, reduction="none")
        >>> out = m(sigmoid_input, target)
        >>> out
        tensor([[2.9266, 1.1963, 1.1087],
                [0.8064, 2.075 , 4.2539]], dtype=oneflow.float32)
        >>> m_sum = flow.nn.BCELoss(weight, reduction="sum")
        >>> out = m_sum(sigmoid_input, target)
        >>> out
        tensor([12.3668], dtype=oneflow.float32)
        >>> m_mean = flow.nn.BCELoss(weight, reduction="mean")
        >>> out = m_mean(sigmoid_input, target)
        >>> out
        tensor([2.0611], dtype=oneflow.float32)
        >>> m_none = flow.nn.BCELoss()
        >>> out = m_none(sigmoid_input, target)
        >>> out
        tensor([1.0306], dtype=oneflow.float32)

    """

    def __init__(self, weight: Tensor = None, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in [
            "none",
            "sum",
            "mean",
            None,
        ], "only 'sum', 'mean' and 'none' supported by now"
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"
        _cross_entropy_loss = flow.experimental.negative(
            target * flow.experimental.log(input)
            + (1 - target) * flow.experimental.log(1 - input)
        )
        if self.weight is not None:
            assert (
                self.weight.shape == input.shape
            ), "The weight shape must be the same as Input shape"
            _weighted_loss = self.weight * _cross_entropy_loss
        else:
            _weighted_loss = _cross_entropy_loss
        if self.reduction == "mean":
            return flow.experimental.mean(_weighted_loss)
        elif self.reduction == "sum":
            return flow.experimental.sum(_weighted_loss)
        else:
            return _weighted_loss


class NLLLoss(Module):
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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import numpy as np

        >>> input = flow.Tensor(
        ... [[-0.1664078, -1.7256707, -0.14690138],
        ... [-0.21474946, 0.53737473, 0.99684894],
        ... [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> m = flow.nn.NLLLoss(reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([ 0.1664, -0.5374, -0.7645], dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor([-1.1355], dtype=oneflow.float32)

        >>> m = flow.nn.NLLLoss(reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor([-0.3785], dtype=oneflow.float32)

    """

    def __init__(
        self, weight=None, ignore_index: int = None, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if weight != None:
            raise ValueError("Argument weight is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"
        self.ignore_index = ignore_index
        self.reduction = reduction

    def nllloss_1d(self, input, target):
        target = flow.F.reshape(target, shape=(target.shape[0], 1))
        res = flow.F.dim_gather(input, target, dim=1)
        res = flow.F.squeeze(res, dim=[1])
        return res

    def forward(self, input, target):
        assert len(input.shape) <= 4
        assert len(target.shape) == len(input.shape) - 1
        input = input.negative()
        if len(input.shape) == 2:
            res = self.nllloss_1d(input, target)
        elif len(input.shape) == 3:
            (b, c, h) = (input.shape[0], input.shape[1], input.shape[2])
            input = flow.F.transpose(input, perm=(0, 2, 1))
            input = input.reshape(shape=[-1, input.shape[2]])
            target = target.flatten()
            res = self.nllloss_1d(input, target)
            res = res.reshape((b, h))
        elif len(input.shape) == 4:
            (b, c, h, w) = (
                input.shape[0],
                input.shape[1],
                input.shape[2],
                input.shape[3],
            )
            input = flow.F.transpose(input, perm=(0, 2, 3, 1))
            input = input.reshape(shape=[-1, input.shape[3]])
            target = target.flatten()
            res = self.nllloss_1d(input, target)
            res = res.reshape((b, h, w))
        else:
            raise NotImplemented
        if self.ignore_index is not None:
            zeros = flow.experimental.zeros(
                size=res.shape, dtype=res.dtype, device=res.device
            )
            condition = flow.experimental.eq(target, self.ignore_index)
            ones = flow.experimental.ones(
                size=condition.shape, dtype=condition.dtype, device=condition.device
            )
            condition = ones.sub(condition).reshape(tuple(res.shape))
            res = flow.experimental.where(condition, res, zeros)
            if self.reduction == "mean":
                res = res.sum()
                reduce_count = condition.argwhere().shape[0]
                res = flow.experimental.mul(res, 1.0 / reduce_count)
        if self.reduction == "none":
            return res
        elif self.reduction == "sum":
            return res.sum()
        else:
            return res.mean()


class KLDivLoss(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss

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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor([-0.9021705, 0.08798598, 1.04686249], dtype=flow.float32)
        >>> target = flow.Tensor([1.22386942, -0.89729659, 0.01615712], dtype=flow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="none", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor([ 1.3514,  0.    , -0.0836], dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="mean", log_target=False)
        >>> out = m(input, target)
        >>> out
        tensor([0.4226], dtype=oneflow.float32)
        >>> m = flow.nn.KLDivLoss(reduction="sum", log_target=True)
        >>> out = m(input, target)
        >>> out
        tensor([5.7801], dtype=oneflow.float32)

    """

    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "Argument reduction only support 'sum'/'mean'/'none'/None for now!"
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.log_target:
            _kl_div_loss = flow.experimental.exp(target) * (target - input)
        else:
            _kl_div_out_loss = target * (flow.experimental.log(target) - input)
            _zeros = flow.experimental.zeros(
                size=_kl_div_out_loss.shape,
                dtype=_kl_div_out_loss.dtype,
                device=_kl_div_out_loss.device,
            )
            _condition = flow.experimental.gt(target, 0)
            _kl_div_loss = flow.experimental.where(_condition, _kl_div_out_loss, _zeros)
        if self.reduction == "mean":
            return flow.experimental.mean(_kl_div_loss)
        elif self.reduction == "sum":
            return flow.experimental.sum(_kl_div_loss)
        else:
            return _kl_div_loss


class MSELoss(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss

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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(
        ... [[-0.02557137, 0.03101675, 1.37493674],
        ... [0.25599439, -1.08372561, -0.21006816]], dtype=flow.float32)
        >>> target = flow.Tensor(
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
        tensor([1.3573], dtype=oneflow.float32)
        >>> m = flow.nn.MSELoss(reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor([8.1436], dtype=oneflow.float32)

    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "Argument reduction only support 'sum'/'mean'/'none'/None for now!"
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean_squared_difference = flow.experimental.square(
            flow.experimental.sub(input, target)
        )
        if self.reduction == "mean":
            return flow.experimental.mean(mean_squared_difference)
        elif self.reduction == "sum":
            return flow.experimental.sum(mean_squared_difference)
        else:
            return mean_squared_difference


class MarginRankingLoss(Module):
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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import numpy as np

        >>> x1 = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=flow.float32)
        >>> x2 = flow.Tensor(np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]), dtype=flow.float32)
        >>> target = flow.Tensor(np.array([[1, -1, 1],[-1, 1, -1], [1, 1, 1]]), dtype=flow.float32)
        >>> m = flow.nn.MarginRankingLoss(margin =1.0, reduction="none")
        >>> out = m(x1, x2, target)
        >>> out
        tensor([[2., 1., 0.],
                [3., 0., 5.],
                [0., 0., 0.]], dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 0.3, reduction="sum")
        >>> out = m(x1, x2, target)
        >>> out
        tensor([8.2], dtype=oneflow.float32)

        >>> m = flow.nn.MarginRankingLoss(margin = 10, reduction="mean")
        >>> out = m(x1, x2, target)
        >>> out
        tensor([8.3333], dtype=oneflow.float32)


    """

    def __init__(self, margin=0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"
        self.reduction = reduction

    def forward(self, input1, input2, target):
        res = flow.experimental.clip(
            flow.experimental.add(
                self.margin,
                flow.experimental.mul(
                    target,
                    flow.experimental.mul(-1, flow.experimental.sub(input1, input2)),
                ),
            ),
            min=0.0,
        )
        if self.reduction == "none":
            return res
        elif self.reduction == "sum":
            return res.sum()
        else:
            return res.mean()


class CTCLoss(Module):
    """The Connectionist Temporal Classification loss.
    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss

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
          :math:`(\\operatorname{sum}(\\text{target\\_lengths}))`,
          where :math:`N = \\text{batch size}` and
          :math:`S = \\text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\\operatorname{sum}(\\text{target\\_lengths}))` form,
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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import numpy as np
        >>> log_probs = np.array(
        ...             [
        ...                 [[-1.1031, -0.7998, -1.5200], [-0.9808, -1.1363, -1.1908]],
        ...                 [[-1.2258, -1.0665, -1.0153], [-1.1135, -1.2331, -0.9671]],
        ...                 [[-1.3348, -0.6611, -1.5118], [-0.9823, -1.2355, -1.0941]],
        ...                 [[-1.3850, -1.3273, -0.7247], [-0.8235, -1.4783, -1.0994]],
        ...                 [[-0.9049, -0.8867, -1.6962], [-1.4938, -1.3630, -0.6547]],
        ...             ]
        ...         ).astype(np.float32)
        >>> log_probs = flow.Tensor(log_probs, dtype=flow.float32)
        >>> targets = flow.Tensor(np.array([[1, 2, 2], [1, 2, 2]]).astype("int32"), dtype=flow.int32)
        >>> input_lengths = flow.Tensor(np.array([5, 5]).astype("int32"), dtype=flow.int32)
        >>> target_lengths = flow.Tensor(np.array([3, 3]).astype("int32"), dtype=flow.int32)
        >>> loss_mean = flow.nn.CTCLoss()
        >>> out = loss_mean(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor([1.1376], dtype=oneflow.float32)
        >>> loss_sum = flow.nn.CTCLoss(blank=0, reduction="sum")
        >>> out = loss_sum(log_probs, targets, input_lengths, target_lengths)
        >>> out
        tensor([6.8257], dtype=oneflow.float32)
        >>> 

    """

    def __init__(
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False
    ) -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self._op = (
            flow.builtin_op("ctc_loss")
            .Input("log_probs")
            .Input("targets")
            .Input("input_lengths")
            .Input("target_lengths")
            .Output("loss")
            .Output("alpha")
            .Attr("blank", int(blank))
            .Attr("zero_infinity", zero_infinity)
            .Build()
        )
        self._xdivy_op = (
            flow.builtin_op("xdivy").Input("x").Input("y").Output("z").Build()
        )
        self.constant = _ConstantBase

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        (loss, _) = self._op(log_probs, targets, input_lengths, target_lengths)
        if self.zero_infinity:
            cond = flow.experimental.eq(
                loss,
                self.constant(
                    size=loss.shape,
                    value=float("inf"),
                    dtype=loss.dtype,
                    device=loss.device,
                )(),
            )
            loss = flow.experimental.where(
                cond,
                flow.experimental.zeros(
                    size=loss.shape, dtype=loss.dtype, device=loss.device
                ),
                loss,
            )
        if self.reduction == "mean":
            return flow.experimental.mean(
                self._xdivy_op(
                    loss,
                    flow.experimental.cast(
                        flow.experimental.clamp(target_lengths, min=1),
                        dtype=log_probs.dtype,
                    ),
                )[0]
            )
        elif self.reduction == "sum":
            return flow.experimental.sum(loss)
        else:
            return loss


class BCEWithLogitsLoss(Module):
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
        size_average (bool, optional) – Deprecated (see :attr:`reduction`). Default: ``True``
        reduce (bool, optional) – Deprecated (see :attr:`reduction`). Default: ``True``
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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import oneflow.compatible.single_client.typing as tp

        >>> input = flow.Tensor([[1.2, 0.2, -0.3], [0.7, 0.6, -2], [0.7, 0.6, -2]], dtype=flow.float32)
        >>> target = flow.Tensor([[0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=flow.float32)
        >>> weight = flow.Tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=flow.float32)
        >>> pos_weight = flow.Tensor([1.2, 1.3, 1.4], dtype=flow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="none")
        >>> out = m(input, target)
        >>> out
        tensor([[2.9266, 1.5552, 1.1087],
                [0.9676, 2.075 , 5.9554],
                [0.9676, 2.075 , 5.9554]], dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="mean")
        >>> out = m(input, target)
        >>> out
        tensor([2.6207], dtype=oneflow.float32)

        >>> m = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="sum")
        >>> out = m(input, target)
        >>> out
        tensor([23.5865], dtype=oneflow.float32)


    """

    def __init__(
        self,
        weight=None,
        size_average: bool = True,
        reduce: bool = True,
        reduction: Optional[str] = "mean",
        pos_weight=None,
    ) -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        if not target.shape == input.shape:
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        _neg_input = flow.experimental.negative(input)
        _max_val = flow.experimental.clip(_neg_input, 0)
        _neg_max_val = flow.experimental.negative(_max_val)
        if self.pos_weight:
            _log_weight = (self.pos_weight - 1) * target + 1
            _loss = (1 - target) * input + _log_weight * (
                flow.experimental.log(
                    flow.experimental.exp(_neg_max_val)
                    + flow.experimental.exp(_neg_input - _max_val)
                )
                + _max_val
            )
        else:
            _loss = (1 - target) * input + _max_val
            _loss += flow.experimental.log(
                flow.experimental.exp(_neg_max_val)
                + flow.experimental.exp(_neg_input - _max_val)
            )
        if self.weight is not None:
            assert (
                self.weight.shape == input.shape
            ), "The weight shape must be the same as Input shape"
            _weighted_loss = self.weight * _loss
        else:
            _weighted_loss = _loss
        if self.reduction == "mean":
            return flow.experimental.mean(_weighted_loss)
        elif self.reduction == "sum":
            return flow.experimental.sum(_weighted_loss)
        else:
            return _weighted_loss


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
