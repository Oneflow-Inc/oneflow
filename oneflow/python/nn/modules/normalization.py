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
from oneflow.python.nn import init
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import Tensor
from typing import Tuple, Union

_shape_t = Union[int, Tuple[int], flow.Size]


@oneflow_export("nn.LayerNorm")
class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__
    
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.
    
    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    
    Args:
        normalized_shape (int or list or flow.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]
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

        import numpy as np
        import oneflow as flow

        input_arr = np.array(
            [
                [
                    [[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],
                    [[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],
                ],
                [
                    [[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],
                    [[1.07541728, 0.11008703], [0.26361224, -0.48663723]],
                ],
            ],
            dtype=np.float32,
        )

        x = flow.Tensor(input_arr)
        m = flow.nn.LayerNorm(2)
        y = m(x)

        # [[[[ 0.99997395 -0.99997395]
        # [-0.999947    0.999947  ]]

        # [[-0.99995947  0.9999595 ]
        # [ 0.99998796 -0.99998796]]]

        # [[[-0.9998348   0.99983454]
        # [ 0.9999913  -0.9999913 ]]

        # [[ 0.99997866 -0.99997854]
        # [ 0.9999645  -0.9999645 ]]]]

    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        self.epsilon = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = flow.nn.Parameter(flow.Tensor(*self.normalized_shape))
            self.bias = flow.nn.Parameter(flow.Tensor(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
        # An integer specifies which axis to normalize at first, defaults to 1.
        self.begin_norm_axis = 1
        # An integer specifies which axis params at, defaults to -1 in 'NCHW' format,-2 in 'NHWC' format
        self.begin_params_axis = -1

        self._op = (
            flow.builtin_op("layer_norm")
            .Input("x")
            .Output("y")
            .Output("mean")
            .Output("inv_variance")
            .Attr("center", False)
            .Attr("scale", False)
            .Attr("begin_params_axis", self.begin_params_axis)
            .Attr("epsilon", self.epsilon)
            .Build()
        )

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        assert len(x.shape) > len(
            self.normalized_shape
        ), "Input tensor dim must greater than normalized dim!"
        self.begin_norm_axis = len(x.shape) - len(self.normalized_shape)
        res = self._op(x, begin_norm_axis=self.begin_norm_axis)[0]
        return res

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
