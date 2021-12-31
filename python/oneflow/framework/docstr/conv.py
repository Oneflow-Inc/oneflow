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
    oneflow._C.conv1d,
    r"""
    conv1d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html?highlight=conv1d

    Applies a 1D convolution over an input signal composed of several input
    planes.

    See :class:`~oneflow.nn.Conv1d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iW)`
        weight: quantized filters of shape :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , iW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out_channels})`. The tensor type must be `flow.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padW,)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
          number of groups. Default: 1

    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.random.randn(33, 16, 30), dtype=flow.float32)
        >>> filters = flow.tensor(np.random.randn(20, 16, 5), dtype=flow.float32)
        >>> out = flow._C.conv1d(input, filters,stride=[1], padding=[0], dilation=[1], channel_pos="channels_first")
        """,
)
add_docstr(
    oneflow._C.conv2d,
    r"""
    conv2d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=conv2d

    Applies a 2D convolution over an input image composed of several input
    planes.

    See :class:`~oneflow.nn.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out_channels})`. The tensor type must be `flow.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
          number of groups. Default: 1
    
    
        """,
)
add_docstr(
    oneflow._C.conv3d,
    r"""
    conv3d(input, weight, bias=None, stride=[1], padding=[0], dilation=[1], groups=1) -> Tensor

    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html?highlight=conv3d

    Applies a 3D convolution over an input image composed of several input
    planes.


    See :class:`~oneflow.nn.Conv3d` for details and output shape.

    Args:
        input: quantized input tensor of shape
          :math:`(\text{minibatch} , \text{in_channels} , iD , iH , iW)`
        weight: quantized filters of shape
          :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kD , kH , kW)`
        bias: **non-quantized** bias tensor of shape
          :math:`(\text{out_channels})`. The tensor type must be `flow.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in_channels}` should be
          divisible by the number of groups. Default: 1
        
        
    """,
)
