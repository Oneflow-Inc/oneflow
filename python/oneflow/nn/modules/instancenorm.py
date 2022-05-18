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
from oneflow.nn.modules.batchnorm import _NormBase


class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.channel_axis = 1

    def forward(self, x):
        self._check_input_dim(x)
        # if self.training and self.track_running_stats:
        #     if self.num_batches_tracked is not None:
        #         self.num_batches_tracked.add_(1)
        if self.training:
            is_training = True
        else:
            is_training = (self.running_mean is None) and (self.running_var is None)
        shape = list(x.shape)
        b = shape[0]
        c = shape[1]
        shape[1] = b * c
        shape[0] = 1
        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        running_var = self.running_var
        if self.weight is not None:
            weight = self.weight.repeat(b)
        if self.bias is not None:
            bias = self.bias.repeat(b)
        if self.running_mean is not None:
            running_mean = self.running_mean.repeat(b)
        if self.running_var is not None:
            running_var = self.running_var.repeat(b)
        reshape_to_1d = flow.reshape(x, shape)
        normalized_1d_out = flow._C.normalization(
            reshape_to_1d,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            axis=self.channel_axis,
            epsilon=self.eps,
            momentum=self.momentum,
            is_training=is_training,
        )
        if self.track_running_stats:
            self.running_mean = flow.reshape(running_mean, [b, c]).mean(0, False)
        if self.track_running_stats:
            self.running_var = flow.reshape(running_var, [b, c]).mean(0, False)
        reshape_back_to_nd = flow.reshape(normalized_1d_out, list(x.shape))
        return reshape_back_to_nd


class InstanceNorm1d(_InstanceNorm):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.InstanceNorm1d.html.

    Applies Instance Normalization over a 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm1d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm1d` is applied
        on each channel of channeled data like multidimensional time series, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm1d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, L)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> # Without Learnable Parameters
        >>> m = flow.nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = flow.nn.InstanceNorm1d(100, affine=True)
        >>> x = flow.Tensor(np.random.randn(20, 100, 40))
        >>> output = m(x)

    """

    def _check_input_dim(self, input):
        if input.dim() == 2:
            raise ValueError(
                "InstanceNorm1d returns 0-filled tensor to 2D tensor.This is because InstanceNorm1d reshapes inputs to(1, N * C, ...) from (N, C,...) and this makesvariances 0."
            )
        if input.dim() != 3:
            raise ValueError("expected 3D input (got {}D input)".format(input.dim()))


class InstanceNorm2d(_InstanceNorm):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.InstanceNorm2d.html.

    Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm2d` is applied
        on each channel of channeled data like RGB images, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm2d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> # Without Learnable Parameters
        >>> m = flow.nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = flow.nn.InstanceNorm2d(100, affine=True)
        >>> x = flow.Tensor(np.random.randn(20, 100, 35, 45))
        >>> output = m(x)

    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class InstanceNorm3d(_InstanceNorm):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.InstanceNorm3d.html.

    Applies Instance Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size C (where C is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm3d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm3d` is applied
        on each channel of channeled data like 3D models with RGB color, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm3d` usually don't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> # Without Learnable Parameters
        >>> m = flow.nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = flow.nn.InstanceNorm3d(100, affine=True)
        >>> x = flow.Tensor(np.random.randn(20, 100, 35, 45, 10))
        >>> output = m(x)

    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
