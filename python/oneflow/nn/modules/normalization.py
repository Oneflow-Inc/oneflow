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
import warnings
from typing import Tuple, Union

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn import init
from oneflow.nn.modules.module import Module

_shape_t = Union[int, Tuple[int], flow._oneflow_internal.Size]


class GroupNorm(Module):
    """
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\\gamma` and :math:`\\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.GroupNorm.html.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\\text{num_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.random.randn(20, 6, 10, 10))
        >>> # Separate 6 channels into 3 groups
        >>> m = flow.nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = flow.nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = flow.nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    
"""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ) -> None:
        super().__init__()
        assert num_groups > 0, "The num_groups must larger than zero"
        assert num_channels > 0, "The num_channels must larger than zero"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = flow.nn.Parameter(flow.Tensor(num_channels))
            self.bias = flow.nn.Parameter(flow.Tensor(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            flow.nn.init.ones_(self.weight)
            flow.nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        assert (
            len(input.shape) >= 3
        ), "The dimensions of input tensor must larger than 2"
        assert (
            input.shape[1] == self.num_channels
        ), "The channels of input tensor must equal num_channels"

        if input.is_cuda:
            return flow._C.group_norm(
                input, self.weight, self.bias, self.affine, self.num_groups, self.eps
            )
        else:
            origin_shape = input.shape
            reshape_to_1d = flow.reshape(
                input, shape=[origin_shape[0], self.num_groups, -1]
            )
            mean = flow.mean(reshape_to_1d, dim=2, keepdim=True)
            variance = flow.var(reshape_to_1d, dim=2, unbiased=False, keepdim=True)
            normalized = (reshape_to_1d - mean) / flow.sqrt(variance + self.eps)
            normalized = flow.reshape(
                normalized, shape=[origin_shape[0], self.num_channels, -1]
            )
            if self.weight is not None:
                normalized = normalized * self.weight.reshape(1, self.num_channels, 1)
            if self.bias is not None:
                normalized = normalized + self.bias.reshape(1, self.num_channels, 1)
            res = flow.reshape(normalized, shape=tuple(input.shape))
            return res

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(
            **self.__dict__
        )


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    assert len(input.shape) > len(
        normalized_shape
    ), "Input tensor dim must greater than normalized dim!"
    begin_norm_axis = len(input.shape) - len(normalized_shape)
    begin_params_axis = len(input.shape) - len(normalized_shape)

    elementwise_affine = True if (weight is not None and bias is not None) else False

    for i in range(0, len(normalized_shape)):
        if input.shape[i + begin_params_axis] != normalized_shape[i]:
            raise RuntimeError(
                f"Given normalized_shape={normalized_shape}, expected input with shape [*, {str(normalized_shape)[1:-1]}], but got input of size {input.shape}"
            )

    input_device_type = input.device.type if input.is_local else input.placement.type
    if input_device_type == "cpu":
        reduce_axis = []
        for dim in range(len(input.shape)):
            if dim >= begin_norm_axis:
                reduce_axis.append(dim)
        mean = input.mean(dim=reduce_axis, keepdim=True)
        variance = input.var(dim=reduce_axis, unbiased=False, keepdim=True)
        params_shape = input.shape[begin_params_axis:]
        if len(mean.shape) == 1:
            nd_params_shape = [1] * len(input.shape)
            nd_params_shape[begin_norm_axis] = params_shape[0]
            mean = flow.reshape(mean, shape=nd_params_shape)
            variance = flow.reshape(variance, nd_params_shape)
            if weight is not None and params_shape[0] == weight.nelement():
                weight = flow.reshape(weight, shape=nd_params_shape)
            if bias is not None and params_shape[0] == bias.nelement():
                bias = flow.reshape(bias, shape=nd_params_shape)
        elif len(mean.shape) == len(input.shape):
            pass
        else:
            raise ValueError(
                "shape of mean and variance should be 1D or has number of axes and x's"
            )
        variance += eps
        normalized = (input - mean) * variance.rsqrt()
        if elementwise_affine:
            normalized = normalized * weight + bias
        return normalized
    else:
        if elementwise_affine:
            res = flow._C.layer_norm_affine(
                input,
                weight,
                bias,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        else:
            res = flow._C.layer_norm(
                input,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                epsilon=eps,
            )
        return res


class LayerNorm(Module):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or oneflow.Size): input shape from an expected input of size

            .. math::
                [* \\times \\text{normalized_shape}[0] \\times \\text{normalized_shape}[1] \\times \\ldots \\times \\text{normalized_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will

            normalize over the last dimension which is expected to be of that specific size.

        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input_arr = np.array(
        ...     [
        ...         [
        ...             [[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],
        ...             [[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],
        ...         ],
        ...         [
        ...             [[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],
        ...             [[1.07541728, 0.11008703], [0.26361224, -0.48663723]],
        ...         ],
        ...     ],
        ...     dtype=np.float32,
        ... )

        >>> x = flow.Tensor(input_arr)
        >>> m = flow.nn.LayerNorm(2)
        >>> y = m(x).numpy()
        >>> y
        array([[[[ 0.99997395, -0.99997395],
                 [-0.999947  ,  0.999947  ]],
        <BLANKLINE>
                [[-0.99995965,  0.9999595 ],
                 [ 0.99998784, -0.99998784]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[-0.9998348 ,  0.99983466],
                 [ 0.9999914 , -0.9999914 ]],
        <BLANKLINE>
                [[ 0.9999785 , -0.9999785 ],
                 [ 0.9999646 , -0.9999646 ]]]], dtype=float32)

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = flow.nn.Parameter(flow.Tensor(*self.normalized_shape))
            self.bias = flow.nn.Parameter(flow.Tensor(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


class RMSLayerNorm(Module):
    """
    Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
    
    T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    
    Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    
    w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    
    half-precision inputs is done in fp32.

    Args:
        hidden_size (int): number of features in the hidden state
        eps: a value added to the denominator for numerical stability. Default: 1e-6

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    
    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.randn(2, 4, 3)
        >>> m = flow.nn.RMSLayerNorm(3)
        >>> y = m(x)
        >>> y.size()
        oneflow.Size([2, 4, 3])

    """

    def __init__(self, hidden_size, eps=1e-6):
        warnings.warn(
            f"nn.RMSLayerNorm has been deprecated. Please use nn.RMSNorm instead."
        )

        super().__init__()
        self.weight = flow.nn.Parameter(flow.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return flow._C.rms_layer_norm(hidden_states, self.weight, self.variance_epsilon)


class RMSNorm(Module):
    """Applies Root Mean Square Layer Normalization over a mini-batch of inputs as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \\frac{x}{\\mathrm{RMS}[x]} \\mathrm{weight},\\text{ where }\\mathrm{RMS}[x] = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} x^{2}}

    There is no bias and no subtraction of mean with RMS Layer Normalization, 
    and it only scales and doesn't shift.

    The root mean squre are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\weight` is learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Like Layer Normalization, Root Mean Square Layer Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or oneflow.Size): input shape from an expected input of size

            .. math::
                [* \\times \\text{normalized_shape}[0] \\times \\text{normalized_shape}[1] \\times \\ldots \\times \\text{normalized_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will

            normalize over the last dimension which is expected to be of that specific size.

        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights).
            Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input_arr = np.array(
        ...     [
        ...         [
        ...             [[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],
        ...             [[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],
        ...         ],
        ...         [
        ...             [[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],
        ...             [[1.07541728, 0.11008703], [0.26361224, -0.48663723]],
        ...         ],
        ...     ],
        ...     dtype=np.float32,
        ... )

        >>> x = flow.Tensor(input_arr, device="cuda")
        >>> m = flow.nn.RMSNorm(2).to(device="cuda")
        >>> y = m(x).numpy()
        >>> y
        array([[[[-0.21632987, -1.3975569 ],
                 [-1.127044  ,  0.8541454 ]],
        <BLANKLINE>
                [[-1.2975204 , -0.5625112 ],
                 [ 1.3711083 ,  0.34648165]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[-1.2388322 , -0.6820876 ],
                 [ 0.16959298, -1.4040003 ]],
        <BLANKLINE>
                [[ 1.4068495 ,  0.14401469],
                 [ 0.6735778 , -1.2434478 ]]]], dtype=float32)

    """

    _constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = flow.nn.Parameter(
                flow.ones(*self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return flow._C.rms_norm(x, self.weight, self.normalized_shape, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
