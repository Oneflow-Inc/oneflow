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

from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
import oneflow._oneflow_internal as oneflow_api


class BatchNormalization(Module):
    def __init__(
        self, axis: int = -1, epsilon: float = 1e-5,
    ):
        super(BatchNormalization, self).__init__()
        self.axis = axis
        self.epsilon = epsilon
        self._op = (
            flow.builtin_op("normalization")
            .Input("x")
            .Input("moving_mean")
            .Input("moving_variance")
            .Input("gamma")
            .Input("beta")
            .Output("y")
            .Attr("axis", axis)
            .Attr("epsilon", epsilon)
            .Attr("momentum", 0.0)  # momentum is not used here
            .Attr("training", False)
            .Build()
        )

    def forward(self, x, mean, variance, weight=None, bias=None):
        assert self.axis >= -len(x.shape) and self.axis < len(x.shape)
        if self.axis < 0:
            self.axis += len(x.shape)

        params_shape = [x.shape[self.axis]]
        if x.device == flow.device("cpu"):
            if len(mean.shape) == 1:
                nd_params_shape = [1] * len(x.shape)
                nd_params_shape[self.axis] = params_shape[0]
                mean = mean.reshape(shape=nd_params_shape)
                variance = variance.reshape(shape=nd_params_shape)

                if weight and params_shape[0] == weight.nelemenet():
                    weight = weight.reshape(shape=nd_params_shape)
                if bias and params_shape[0] == bias.nelemenet():
                    bias = bias.reshape(shape=nd_params_shape)
            elif len(mean.shape) == len(x.shape):
                pass
            else:
                raise ValueError(
                    "shape of mean and variance should be 1D or has number of axes and x's"
                )

            variance += self.epsilon
            normalized = (x - mean) * variance.rsqrt()
            affined = normalized

            if weight:
                affined = affined * weight
            if bias:
                affined = affined + bias
            return affined
        else:
            if weight is None:
                weight = flow.experimental.ones(size=params_shape, dtype=params_dtype)
            if bias is None:
                bias = flow.experimental.zeros(size=params_shape, dtype=params_dtype)
            res = self._op(x, mean, variance, weight, bias)[0]
            return res


@oneflow_export("batch_normalization")
@experimental_api
def batch_normalization_op(
    x, mean, variance, weight=None, bias=None, axis=1, epsilon=1e-5
):
    return BatchNormalization(axis, epsilon)(x, mean, variance, weight, bias)


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
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
            self.weight = flow.nn.Parameter(
                flow.Tensor(num_features).normal_(mean=0.0, std=1.0)
            )
            self.bias = flow.nn.Parameter(
                flow.Tensor(num_features).normal_(mean=0.0, std=1.0)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", flow.Tensor(num_features).fill_(0.0))
            self.register_buffer("running_var", flow.Tensor(num_features).fill_(1.0))
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
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._training_op = (
            flow.builtin_op("normalization")
            .Input("x")
            .Input("moving_mean")
            .Input("moving_variance")
            .Input("gamma")
            .Input("beta")
            .Attr("axis", 1)
            .Attr("epsilon", eps)
            .Attr("momentum", momentum)
            .Output("y")
            .Output("mean")
            .Output("inv_variance")
            .Attr("training", True)
            .Build()
        )
        self._testing_op = (
            flow.builtin_op("normalization")
            .Input("x")
            .Input("moving_mean")
            .Input("moving_variance")
            .Input("gamma")
            .Input("beta")
            .Attr("axis", 1)
            .Attr("epsilon", eps)
            .Attr("momentum", momentum)
            .Output("y")
            .Attr("training", False)
            .Build()
        )

    def forward(self, x):
        self._check_input_dim(x)

        if x.device == flow.device("cpu"):
            if self.training:
                reduce_axis = []
                for dim in range(len(x.shape)):
                    if dim != 1:
                        reduce_axis.append(dim)
                mean = flow.experimental.reduce_mean(
                    x, axis=reduce_axis, keepdims=False
                )
                variance = flow.experimental.reduce_variance(
                    x, axis=reduce_axis, keepdims=False
                )

                running_mean = (
                    self.momentum * self.running_mean + (1 - self.momentum) * mean
                )
                running_var = (
                    self.momentum * self.running_var + (1 - self.momentum) * variance
                )

                if self.track_running_stats:
                    self.__dict__.get("_buffers")["running_mean"] = running_mean
                    self.__dict__.get("_buffers")["running_var"] = running_var
                else:
                    del self.__dict__["running_mean"]
                    del self.__dict__["running_var"]
                    self.register_parameter("running_mean", running_mean)
                    self.register_parameter("running_var", running_var)

                # TODO: update running_mean and running_var should use below codes(rather than upper), but raise exception:
                # TypeError: cannot assign '<class 'oneflow._oneflow_internal.LocalTensor'>' as buffer 'running_mean' (Tensor or None expected)
                # self.__setattr__("running_mean", running_mean)
                # self.__setattr__("running_var", running_var)

                return flow.experimental.batch_normalization(
                    x=x,
                    mean=mean,
                    variance=variance,
                    weight=self.weight,
                    bias=self.bias,
                    axis=1,
                    epsilon=self.eps,
                )

            else:
                return flow.experimental.batch_normalization(
                    x=x,
                    mean=self.running_mean,
                    variance=self.running_var,
                    weight=self.weight,
                    bias=self.bias,
                    axis=1,
                    epsilon=self.eps,
                )

        else:
            if self.training:
                res = self._training_op(
                    x, self.running_mean, self.running_var, self.weight, self.bias
                )[0]
            else:
                res = self._testing_op(
                    x, self.running_mean, self.running_var, self.weight, self.bias
                )[0]
            return res


@oneflow_export("nn.BatchNorm1d")
@experimental_api
class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

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

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.randn(20, 100))
        m = flow.nn.BatchNorm1d(100)
        y = m(x)

    """

    def _check_input_dim(self, input):
        if input.ndim != 2 and input.ndim != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.ndim)
            )


@oneflow_export("nn.BatchNorm2d")
@experimental_api
class BatchNorm2d(_BatchNorm):
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

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.randn(4, 2, 8, 3))
        m = flow.nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1)
        y = m(x)

    """

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.ndim()))
