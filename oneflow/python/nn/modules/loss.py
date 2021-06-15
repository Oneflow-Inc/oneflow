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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.constant import _ConstantBase


@oneflow_export("nn.CrossEntropyLoss")
@experimental_api
class CrossEntropyLoss(Module):
    r"""This criterion combines :class:`~flow.nn.LogSoftmax` and :class:`~flow.nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`;

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(
        ...    [[-0.1664078, -1.7256707, -0.14690138],
        ...        [-0.21474946, 0.53737473, 0.99684894],
        ...        [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> out = flow.nn.CrossEntropyLoss(reduction="none")(input, target)
        >>> print(out.numpy())
        [0.80199665 1.1166505  0.35826024]
        >>> out_sum = flow.nn.CrossEntropyLoss(reduction="sum")(input, target)
        >>> print(out_sum.numpy())
        [2.2769072]
        >>> out_mean = flow.nn.CrossEntropyLoss(reduction="mean")(input, target)
        >>> print(out_mean.numpy())
        [0.75896907]


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
        if ignore_index is not None:
            raise ValueError("Argument ignore_index is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.reduction = reduction
        self._op = (
            flow.builtin_op("sparse_softmax_cross_entropy")
            .Input("prediction")
            .Input("label")
            .Output("prob")
            .Output("out")
            .Build()
        )
        self._transpose_op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", [])
            .Build()
        )

    def forward(self, input, target):
        assert len(input.shape) <= 4
        assert len(target.shape) == len(input.shape) - 1
        input_shape_len = len(input.shape)
        if input_shape_len == 3:
            b, c, h = input.shape[0], input.shape[1], input.shape[2]
            input = self._transpose_op(input, perm=(0, 2, 1))[0]
            input = input.reshape(shape=[-1, input.shape[2]])
            target = target.flatten()
        elif input_shape_len == 4:
            b, c, h, w = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
            input = self._transpose_op(input, perm=(0, 2, 3, 1))[0]
            input = input.reshape(shape=[-1, input.shape[3]])
            target = target.flatten()
        elif input_shape_len >= 5:
            raise NotImplemented

        prob, out = self._op(input, target, depth=input.shape[len(input.shape) - 1])
        if self.reduction == "mean":
            return flow.experimental.mean(out)
        elif self.reduction == "sum":
            return flow.experimental.sum(out)
        else:
            if input_shape_len == 4:
                out = out.reshape((b, h, w))
            return out


@oneflow_export("nn.NLLLoss")
@experimental_api
class NLLLoss(Module):
    r""" The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`;

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \mathbb{1},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{N} l_n, &
            \text{if reduction} = \text{`mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{`sum'.}
        \end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import numpy as np

        >>> input = flow.Tensor(
        ... [[-0.1664078, -1.7256707, -0.14690138],
        ... [-0.21474946, 0.53737473, 0.99684894],
        ... [-1.135804, -0.50371903, 0.7645404]], dtype=flow.float32)
        >>> target = flow.Tensor(np.array([0, 1, 2]), dtype=flow.int32)
        >>> m = flow.nn.NLLLoss(reduction="none")
        >>> out = m(input, target).numpy()
        >>> print(out)
        [ 0.1664078  -0.53737473 -0.7645404 ]

        >>> m = flow.nn.NLLLoss(reduction="sum")
        >>> out = m(input, target).numpy()
        >>> print(out)
        [-1.1355073]

        >>> m = flow.nn.NLLLoss(reduction="mean")
        >>> out = m(input, target).numpy()
        >>> print(out)
        [-0.37850246]

    """

    def __init__(
        self, weight=None, ignore_index: int = None, reduction: str = "mean",
    ) -> None:
        super().__init__()
        if weight != None:
            raise ValueError("Argument weight is not supported yet")
        if ignore_index != None:
            raise ValueError("Argument ignore_index is not supported yet")
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.reduction = reduction
        self._dim_gather_op = (
            flow.builtin_op("dim_gather")
            .Input("input")
            .Input("index")
            .Output("output")
            .Attr("dim", 1)
            .Build()
        )
        self._transpose_op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", [])
            .Build()
        )

    def nllloss_1d(self, input, target):
        target = flow.experimental.reshape(target, (target.shape[0], 1))
        res = self._dim_gather_op(input, target)[0]
        res = flow.experimental.squeeze(res, dim=[1])
        return res

    def forward(self, input, target):
        assert len(input.shape) <= 4
        assert len(target.shape) == len(input.shape) - 1
        input = input.negative()
        if len(input.shape) == 2:
            res = self.nllloss_1d(input, target)
        elif len(input.shape) == 3:
            b, c, h = input.shape[0], input.shape[1], input.shape[2]
            input = self._transpose_op(input, perm=(0, 2, 1))[0]
            input = input.reshape(shape=[-1, input.shape[2]])
            target = target.flatten()
            res = self.nllloss_1d(input, target)
            res = res.reshape((b, h))
        elif len(input.shape) == 4:
            b, c, h, w = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
            input = self._transpose_op(input, perm=(0, 2, 3, 1))[0]
            input = input.reshape(shape=[-1, input.shape[3]])
            target = target.flatten()
            res = self.nllloss_1d(input, target)
            res = res.reshape((b, h, w))
        else:
            raise NotImplemented

        if self.reduction == "none":
            return res
        elif self.reduction == "sum":
            return res.sum()
        else:
            return res.mean()


@oneflow_export("nn.MSELoss")
@experimental_api
class MSELoss(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss

    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

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

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
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
        >>> print(out.numpy())
        [[2.266468   0.50750285 0.61121327]
         [0.55887264 4.082267   0.1172941 ]]
        >>> m = flow.nn.MSELoss(reduction="mean")
        >>> out = m(input, target)
        >>> print(out.numpy())
        [1.3572696]
        >>> m = flow.nn.MSELoss(reduction="sum")
        >>> out = m(input, target)
        >>> print(out.numpy())
        [8.143618]

    """

    def __init__(
        self, reduction: str = "mean", size_average: bool = True, reduce: bool = True
    ) -> None:
        super().__init__()
        if size_average is False:
            raise ValueError("Argument size_average is not supported yet")
        if reduce is False:
            raise ValueError("Argument reduce is not supported yet")
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
            # Do no reduction
            return mean_squared_difference


@oneflow_export("nn.MarginRankingLoss")
@experimental_api
class MarginRankingLoss(Module):
    r"""Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        \text{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \text{margin})

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
        
        >>> import oneflow.experimental as flow
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


@oneflow_export("nn.CTCLoss")
@experimental_api
class CTCLoss(Module):
    r"""The Connectionist Temporal Classification loss.
    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

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
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
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

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
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
        self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False,
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
        loss, _ = self._op(log_probs, targets, input_lengths, target_lengths)
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
