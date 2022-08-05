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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow._C.deconv1d,
    r"""
    conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

    Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called “deconvolution”.
    
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.conv_transpose1d.html

    See :class:`~oneflow.nn.ConvTranspose1d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iW)`
        weight: filters of shape :math:`(\text{in_channels} , \frac{\text{out_channels}}{\text{groups}} , kW)`
        bias: optional bias of shape :math:`(\text{out_channels})`. Default: None.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sW,)`. Default: 1
        padding: `dilation * (kernel_size - 1) - padding` zero-padding will be added to both sides of each dimension in the input. Can be a single number or a tuple `(padW,)`. Default: 0
        output_padding: additional size added to one side of each dimension in the output shape. Can be a single number or a tuple `(out_padW)`. Default: 0
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
          number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dW,)`. Default: 1

    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        
        >>> inputs = flow.randn(20, 16, 50)
        >>> weights = flow.randn(16, 33, 5)
        >>> outputs = F.conv_transpose1d(inputs, weights)
        """,
)
add_docstr(
    oneflow._C.deconv2d,
    r"""
    conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

    Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.conv_transpose3d.html

    See :class:`~oneflow.nn.ConvTranspose2d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
        weight: filters of shape :math:`(\text{in_channels} , \frac{\text{out_channels}}{\text{groups}} , kH , kW)`
        bias: optional bias of shape :math:`(\text{out_channels})`. Default: None.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: `dilation * (kernel_size - 1) - padding` zero-padding will be added to both sides of each dimension in the input. Can be a single number or a tuple `(padH, padW)`. Default: 0
        output_padding: additional size added to one side of each dimension in the output shape. Can be a single number or a tuple `(out_padH, out_padW)`. Default: 0
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
          number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
    
    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        
        >>> inputs = flow.randn(1, 4, 5, 5)
        >>> weights = flow.randn(4, 8, 3, 3)
        >>> outputs = F.conv_transpose2d(inputs, weights, padding=1)
        """,
)
add_docstr(
    oneflow._C.deconv3d,
    r"""
    conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor

    Applies a 3D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.conv_transpose3d.html

    See :class:`~oneflow.nn.ConvTranspose3d` for details and output shape.

    Args:
        input: input tensor of shape
          :math:`(\text{minibatch} , \text{in_channels} , iT , iH , iW)`
        weight: filters of shape
          :math:`(\text{in_channels} , \frac{\text{out_channels}}{\text{groups}} , kT , kH , kW)`
        bias: optional bias of shape :math:`(\text{out_channels})`. Default: None.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: `dilation * (kernel_size - 1) - padding` zero-padding will be added to both sides of each dimension in the input. Can be a single number or a tuple `(padT, padH, padW)`. Default: 0
        output_padding: additional size added to one side of each dimension in the output shape. Can be a single number or a tuple `(out_padT, out_padH, out_padW)`. Default: 0
        groups: split input into groups, :math:`\text{in_channels}` should be
          divisible by the number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dT, dH, dW)`. Default: 1
        
    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        
        >>> inputs = flow.randn(20, 16, 50, 10, 20)
        >>> weights = flow.randn(16, 33, 3, 3, 3)
        >>> outputs = F.conv_transpose3d(inputs, weights)
    """,
)
