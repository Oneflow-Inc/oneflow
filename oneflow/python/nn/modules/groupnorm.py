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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


@oneflow_export("nn.GroupNorm")
@experimental_api
class GroupNorm(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

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
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__()
        assert num_groups > 0, "The num_groups must larger than zero"
        assert num_channels > 0, "The num_channels must larger than zero"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = flow.nn.Parameter(flow.Tensor(1, num_channels, 1))
            self.bias = flow.nn.Parameter(flow.Tensor(1, num_channels, 1))
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
        origin_shape = input.shape
        reshape_to_1d = flow.experimental.reshape(
            input, shape=[origin_shape[0], self.num_groups, -1]
        )
        mean = flow.experimental.mean(reshape_to_1d, dim=2, keepdim=True)
        variance = flow.experimental.var(reshape_to_1d, dim=2, keepdim=True)
        normalized = (reshape_to_1d - mean) / flow.experimental.sqrt(
            variance + self.eps
        )
        normalized = flow.experimental.reshape(
            normalized, shape=[origin_shape[0], self.num_channels, -1]
        )
        if self.weight:
            normalized = normalized * self.weight
        if self.bias:
            normalized = normalized + self.bias
        res = flow.experimental.reshape(normalized, shape=tuple(input.shape))

        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
