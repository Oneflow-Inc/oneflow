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
from oneflow.autograd import Function


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

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
            self.register_buffer("running_mean", flow.zeros(num_features))
            self.register_buffer("running_var", flow.ones(num_features))
            self.register_buffer("num_batches_tracked", flow.tensor(0, dtype=flow.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

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
        if self.track_running_stats:
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if not num_batches_tracked_key in state_dict:
                if self.running_mean.is_global:
                    sbp = self.running_mean.sbp
                    placement = self.running_mean.placement
                    state_dict[num_batches_tracked_key] = flow.tensor(
                        0, dtype=flow.long
                    ).to_global(sbp=sbp, placement=placement)
                else:
                    state_dict[num_batches_tracked_key] = flow.tensor(
                        0, dtype=flow.long
                    )
        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )


class _BatchNorm(_NormBase):
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

    def forward(self, x):
        self._check_input_dim(x)
        exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        if self.training:
            is_training = True
        else:
            is_training = (self.running_mean is None) and (self.running_var is None)
        # NOTE(lixiang): If it is training mode, pass running_mean and running_var directly to the functor layer.
        return flow._C.normalization(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            axis=self.channel_axis,
            epsilon=self.eps,
            momentum=exponential_average_factor,
            is_training=is_training,
        )


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
    via the biased estimator, equivalent to `oneflow.var(input, unbiased=False)`.

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
    via the biased estimator, equivalent to `oneflow.var(input, unbiased=False)`.

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
        
        >>> x = flow.Tensor(np.random.randn(4, 2, 8, 3))
        >>> m = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
        >>> y = m(x)

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


class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `oneflow.var(input, unbiased=False)`.

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

        >>> x = flow.Tensor(np.random.randn(3, 2, 5, 8, 4))
        >>> m = flow.nn.BatchNorm3d(num_features=2, eps=1e-5, momentum=0.1)
        >>> y = m(x)
        >>> y.size()
        oneflow.Size([3, 2, 5, 8, 4])

    """

    def _check_input_dim(self, input):
        if input.ndim != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.ndim))


global_eps = 0.1
global_momentum = 0.1
global_world_size = 1
global_axis = 1


class SyncBatchNormFunction(flow.autograd.Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var):
        assert input.is_local, "SyncBatchNorm does not support global tensor as input."

        if not input.is_contiguous():
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and global_world_size < 2:
            raise ValueError(
                "Expected more than 1 value per channel when training, got input size {}".format(
                    size
                )
            )

        num_channels = input.shape[global_axis]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = flow._C.batch_norm_stats(input, global_axis, global_eps)

            count = flow.full(
                (1,),
                input.numel() // input.size(global_axis),
                dtype=mean.dtype,
                device=mean.device,
            )

            # C, C, 1 -> (2C + 1)
            combined = flow.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = flow.zeros(
                2 * num_channels + 1, dtype=input.dtype, device=input.device
            )

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # world_size * (2C + 1)
        combined_size = combined.numel()
        combined_flat = flow.empty(
            global_world_size,
            combined_size,
            dtype=combined.dtype,
            device=combined.device,
        )
        flow.comm.all_gather_into_tensor(combined_flat, combined)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = flow.split(combined_flat, num_channels, dim=1)

        # remove stats from empty inputs
        mask = count_all.squeeze(-1) >= 1
        count_all = count_all[mask]
        mean_all = mean_all[mask]
        invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        mean, invstd = flow._C.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            global_momentum,
            global_eps,
            count_all.view(-1),
        )

        self.save_for_backward(input, weight, mean, invstd, count_all.to(flow.int32))

        # apply element-wise normalization
        if input.numel() > 0:
            return flow._C.batch_norm_elemt(
                input, weight, bias, mean, invstd, global_axis, global_eps
            )
        else:
            return flow.zeros(*(input.shape), dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        channel_axis = 1
        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            if saved_input.dim() == 3:
                channel_axis = 2
            elif saved_input.dim() == 4:
                channel_axis = 3
            elif saved_input.dim() == 5:
                channel_axis = 4

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = flow._C.batch_norm_backward_reduce(
            grad_output, saved_input, mean, invstd, channel_axis
        )

        # synchronizing stats used to calculate input gradient.
        num_channels = sum_dy.shape[0]
        combined = flow.cat([sum_dy, sum_dy_xmu], dim=0)
        flow.comm.all_reduce(combined)
        sum_dy, sum_dy_xmu = flow.split(combined, num_channels)

        # backward pass for gradient calculation
        grad_input = flow._C.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor,
            channel_axis,
        )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        return grad_input, grad_weight, grad_bias, None, None


class SyncBatchNorm(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `oneflow.var(input, unbiased=False)`.

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

    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.

    Currently :class:`SyncBatchNorm` only supports
    :class:`~oneflow.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`oneflow.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
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
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.

    Examples::

        >>> import oneflow as flow
        
        >>> bn = flow.nn.BatchNorm2d(100)
        >>> sync_bn = flow.nn.SyncBatchNorm.convert_sync_batchnorm(bn).cuda()
        >>> input = flow.randn(20, 100, 35, 45, device="cuda")
        >>> output = sync_bn(input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )
        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            if input.dim() == 3:
                self.channel_axis = 2
            elif input.dim() == 4:
                self.channel_axis = 3
            elif input.dim() == 5:
                self.channel_axis = 4

    def _check_non_zero_input_channels(self, input):
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = bn_training and self.training
        if need_sync:
            need_sync = flow.env.get_world_size() > 1

        # # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return flow._C.normalization(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                axis=self.channel_axis,
                epsilon=self.eps,
                momentum=exponential_average_factor,
                is_training=bn_training,
            )
        else:
            assert bn_training
            global global_eps
            global global_momentum
            global global_world_size
            global global_axis
            global_eps = self.eps
            global_momentum = exponential_average_factor
            global_world_size = flow.env.get_world_size()
            global_axis = self.channel_axis
            assert (
                self.track_running_stats
            ), "`track_running_stats` should be True when using SyncBatchNorm."
            return SyncBatchNormFunction.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
            )

    @classmethod
    def convert_sync_batchnorm(cls, module):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`oneflow.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers

        Returns:
            The original :attr:`module` with the converted :class:`oneflow.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`oneflow.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> import oneflow as flow

            >>> module = flow.nn.Sequential( flow.nn.Linear(20, 100), flow.nn.BatchNorm1d(100)).cuda()
            >>> sync_bn_module = flow.nn.SyncBatchNorm.convert_sync_batchnorm(module)

        """
        module_output = module
        if isinstance(module, flow.nn.modules.batchnorm._BatchNorm):
            module_output = flow.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            if module.affine:
                with flow.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child))
        del module
        return module_output


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
