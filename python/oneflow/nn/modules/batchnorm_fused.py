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
from typing import Union
import os

import oneflow as flow
from oneflow.nn.modules.module import Module


class _FusedNormBase(Module):
    """Common base of _FusedBatchNorm"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = flow.nn.Parameter(flow.Tensor(num_features))
            self.bias = flow.nn.Parameter(flow.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", flow.Tensor(num_features))
            self.register_buffer("running_var", flow.Tensor(num_features))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.fill_(0)
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            flow.nn.init.ones_(self.weight)
            flow.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return "num_features={num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )


class _FusedBatchNorm(_FusedNormBase):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.channel_axis = 1

    def forward(self, x, addend=None):
        self._check_input_dim(x)

        if self.training:
            is_training = True
        else:
            is_training = (self.running_mean is None) and (self.running_var is None)
        return flow._C.normalization_add_relu(
            x,
            addend if addend is not None else None,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            axis=self.channel_axis,
            epsilon=self.eps,
            momentum=self.momentum,
            is_training=is_training,
        )


class FusedBatchNorm1d(_FusedBatchNorm):
    """Applies Fused Batch Normalization over a 2D or 3D input, the formula is: 
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    The formula of Batch Normalization is: 

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\\gamma` are set
    to 1 and the elements of :math:`\\beta` are set to 0. The standard-deviation is calculated
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
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
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
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.Tensor(np.random.randn(20, 100)).to("cuda") # FusedBatchNorm support in GPU currently. 
        >>> m = flow.nn.FusedBatchNorm1d(num_features=100, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """

    def _check_input_dim(self, input):
        if input.ndim != 2 and input.ndim != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.ndim)
            )


class FusedBatchNorm2d(_FusedBatchNorm):
    """Applies Fused Batch Normalization over a 4D input, the formula is: 
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    The formula of Batch Normalization is: 

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\\gamma` are set
    to 1 and the elements of :math:`\\beta` are set to 0. The standard-deviation is calculated
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
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
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

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.Tensor(np.random.randn(4, 2, 8, 3)).to("cuda") # FusedBatchNorm support in GPU currently. 
        >>> m = flow.nn.FusedBatchNorm2d(num_features=2, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            self.channel_axis = 3

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.ndim))


class FusedBatchNorm3d(_FusedBatchNorm):
    r"""Applies Fused Batch Normalization over a 5D input, the formula is: 
    
    .. math:: 

        out = ReLU(BatchNorm(input) + addend)

    The formula of Batch Normalization is: 

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

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
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times     x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
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
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.Tensor(np.random.randn(3, 2, 5, 8, 4)).to("cuda") # FusedBatchNorm support in GPU currently. 
        >>> m = flow.nn.FusedBatchNorm3d(num_features=2, eps=1e-5, momentum=0.1).to("cuda")
        >>> y = m(x, addend=None)

    """

    def _check_input_dim(self, input):
        if input.ndim != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.ndim))


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
