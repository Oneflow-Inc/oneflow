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
import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util


@oneflow_export("nn.BatchNorm2d")
class BatchNorm2d(Module):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=False):
        super().__init__()

        self._axis = 1 # for '`NCHW'` data format. Defaults to 1.
        self._training = affine
        self._epsilon = eps
        self._momentum = momentum
        assert track_running_stats==False, "Not support track_running_stats=True yet!"

        # self._op = (
        #         flow.builtin_op("normalization")
        #         .Name(id_util.UniqueStr("BatchNorm_"))
        #         .Attr("epsilon", self._epsilon)
        #         .Attr("training", self._training)
        #         .Attr("momentum", self._momentum)
        #         .Input("x")
        #         .Input("moving_mean")
        #         .Input("moving_variance")
        #         .Input("gamma")
        #         .Input("beta")
        #         .Output("y")
        #         .Build()
        # )


        self._layers_op = (
                flow.builtin_op("layers.batch_normalization")
                .Input("inputs")
                .Input("axis")
                .Input("momentum")
                .Input("epsilon")
                .Input("center")
                .Input("scale")
                .Input("trainable")
                .Input("training")
                .Input("name")
                .Output("y")
                .Build()
        )



    def forward(self, x):
        # with flow.scope.namespace(id_util.UniqueStr("Moments_")):
        #     mean = flow.math.reduce_mean(x, axis=self._axis, keepdims=False),
        #     variance = flow.math.reduce_variance(x, axis=self._axis, keepdims=False)
        
        # (mean, variance) = flow.nn.moments(x, [2], keepdims=True)
        
        # params_dtype = flow.float32 if x.dtype == flow.float16 else x.dtype
        # params_shape = [x.shape[self._axis]]
        # gamma = flow.constant(1, dtype=params_dtype, shape=params_shape, name="gamma")
        # beta = flow.constant(0, dtype=params_dtype, shape=params_shape, name="beta")
        # res = self._op(x, mean, variance, gamma, beta)[0]

        res = self._layers_op(
            x, self._axis, self._momentum, self._epsilon, 
            True, True, True, True, id_util.UniqueStr("BatchNorm_")
        )[0]
        return res