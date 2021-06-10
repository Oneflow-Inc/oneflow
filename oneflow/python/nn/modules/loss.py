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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


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
        self, weight=None, ignore_index: int = None, reduction: str = "none",
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


@oneflow_export("nn.BCEWithLogitsLoss")
@experimental_api
class BCEWithLogitsLoss(Module):

    r"""This operator combines the `Sigmoid` and `BCELoss` together. For numerical stability,
    we apply some math tricks instead of using `Sigmoid` layer with `BCELoss`.

    The equation is:

    if reduction = "none":

    .. math::

        out = -weight*[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    if reduction = "mean":

    .. math::

        out = -\frac{weight}{n}\sum_{i=1}^n[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    if reduction = "sum":

    .. math::

        out =k -weight*\sum_{i=1}^n[Pos\_weight*y*log\sigma({x}) + (1-y)*log(1-\sigma(x))]

    Args:
        input (oneflow._oneflow_internal.BlobDesc): The input Tensor.
        target (oneflow._oneflow_internal.BlobDesc): The target Tensor.
        weight (remote_blob_util, optional): The manual rescaling weight to the loss. Defaults to None.
        pos_weight (remote_blob_util, optional): The manual rescaling weight to the positive examples. Defaults to None.
        reduction (str, optional): The reduce type, it can be one of "none", "mean", "sum". Defaults to "mean".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow._oneflow_internal.BlobDesc: The result Blob.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import oneflow.typing as tp
        >>> import numpy as np

        >>> np_input = flow.Tensor([[1.2, 0.2, -0.3], [0.7, 0.6, -2], [0.7, 0.6, -2]], dtype=flow.float32)

        >>> np_target = flow.Tensor([[0, 1, 0], [1, 0, 1], [1, 0, 1]],dtype=flow.float32)

        >>> np_weight = flow.Tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]],dtype=flow.float32)

        >>> np_pos_weight = flow.Tensor([1.2, 1.3, 1.4], dtype=flow.float32)

        # >>> m = flow.nn.BCEWithLogitsLoss(weight=np_weight, pos_weight=np_pos_weight, reduction="none")
        # >>> out = m(np_input, np_target).numpy()
        # >>> print(out)
        # [[2.926565, 1.5551611, 1.1087105],
        #  [0.96764666, 2.074976, 5.9553986]]
        [[2.9266 1.5552 1.1087], [0.9676 2.0750 5.9554], [0.9676, 2.0750, 5.9554]]

        >>> m = flow.nn.BCEWithLogitsLoss(weight=np_weight, pos_weight=np_pos_weight, reduction="mean")
        >>> out = m(np_input, np_target).numpy()
        >>> print(out)
        [2.62072]

        >>> m = flow.nn.BCEWithLogitsLoss(weight=np_weight, pos_weight=np_pos_weight, reduction="sum")
        >>> out = m(np_input, np_target).numpy()
        >>> print(out)
        [23.58648]

    """

    def __init__(
        self,
        weight=None,
        pos_weight=None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super().__init__()
        assert reduction in [
            "sum",
            "none",
            "mean",
            None,
        ], "only 'sum', 'mean' and None supported by now"

        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        self._transpose_op = (
            flow.builtin_op("transpose")
            .Input("input")
            .Output("output")
            .Attr("perm", [])
            .Build()
        )

    def forward(self, input, target):
        if len(input.shape) >= 5:
            raise NotImplemented

        _neg_input = flow.experimental.negative(input)
        _max_val = flow.experimental.clip(_neg_input,0)
        _neg_max_val = flow.experimental.negative(_max_val)

        if self.pos_weight:
            assert self.pos_weight.shape[0] == input.shape[-1], (
                "The length of `pos_weight` must be equal to the number of classes. "
                "Found the length of pos_weight {} vs classes {}".format(
                    self.pos_weight.shape[0], input.shape[-1]
                )
            )
            _log_weight = ((self.pos_weight - 1) * target) + 1
            _loss = (1 - target) * input + _log_weight * (
                    flow.experimental.log(
                        flow.experimental.exp(_neg_max_val) + flow.experimental.exp(_neg_input - _max_val)
                    )
                    + _max_val
            )
        else:
            _loss = (1 - target) * input + _max_val
            _loss += flow.experimental.log(
                flow.experimental.exp(_neg_max_val) + flow.experimental.exp(_neg_input - _max_val)
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
            # Do no reduction
            return _weighted_loss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
