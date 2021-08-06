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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.module import Module


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Union[str, flow.device] = None,
        dtype: flow.dtype = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype
        if self.affine:
            self.weight = flow.nn.Parameter(
                flow.Tensor(num_features, device=self.device)
            )
            self.bias = flow.nn.Parameter(flow.Tensor(num_features, device=self.device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", flow.Tensor(num_features, device=self.device)
            )
            self.register_buffer(
                "running_var", flow.Tensor(num_features, device=self.device)
            )
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )

    def forward(self, x):
        if self.dtype is None:
            self.dtype = x.dtype
        if self.device is None:
            self.device = x.device
        self._check_input_dim(x)
        reduce_axis = []
        for dim in range(len(x.shape)):
            if dim != 1:
                reduce_axis.append(dim)
        mean = x.mean(dim=reduce_axis, keepdim=False)
        variance = x.var(dim=reduce_axis, keepdim=False)
        if x.device == flow.device("cpu"):
            if self.training and self.track_running_stats:
                running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * mean
                )
                running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * variance
                )
                self.__setattr__("running_mean", flow.Tensor(running_mean))
                self.__setattr__("running_var", flow.Tensor(running_var))
            else:
                mean = mean if self.running_mean is None else self.running_mean
                variance = variance if self.running_var is None else self.running_var
            axis = 1
            params_shape = [x.shape[axis]]
            weight = self.weight
            bias = self.bias
            if len(mean.shape) == 1:
                nd_params_shape = [1] * len(x.shape)
                nd_params_shape[axis] = params_shape[0]
                mean = mean.reshape(shape=nd_params_shape)
                variance = variance.reshape(shape=nd_params_shape)
                if self.weight and params_shape[0] == self.weight.nelement():
                    weight = self.weight.reshape(shape=nd_params_shape)
                if self.bias and params_shape[0] == self.bias.nelement():
                    bias = self.bias.reshape(shape=nd_params_shape)
            elif len(mean.shape) == len(x.shape):
                pass
            else:
                raise ValueError(
                    "shape of mean and variance should be 1D or has number of axes and x's"
                )
            variance += self.eps
            normalized = (x - mean) * variance.rsqrt()
            affined = normalized
            if self.weight:
                affined = affined * weight
            if self.bias:
                affined = affined + bias
            return affined.to(dtype=self.dtype)
        else:
            res = flow.F.normalization(
                x,
                self.running_mean if self.track_running_stats else mean,
                self.running_var if self.track_running_stats else variance,
                self.weight,
                self.bias,
                axis=1,
                epsilon=self.eps,
                momentum=self.momentum,
                is_training=self.training,
            )
            return res.to(dtype=self.dtype, device=self.device)


class BatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.random.randn(20, 100))
        >>> m = flow.nn.BatchNorm1d(100)
        >>> y = m(x)

    """

    def _check_input_dim(self, input):
        if input.ndim != 2 and input.ndim != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.ndim)
            )


class BatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.random.randn(4, 2, 8, 3))
        >>> m = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
        >>> y = m(x)

    """

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.ndim()))


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
