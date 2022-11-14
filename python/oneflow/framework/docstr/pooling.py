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
    oneflow._C.adaptive_avg_pool1d,
    """
    adaptive_avg_pool1d(input, output_size) -> Tensor

    Applies a 1D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~oneflow.nn.AdaptiveAvgPool1d` for details and output shape.

    Args:
        input: the input tensor
        output_size: the target output size (single integer)

    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([[[ 0.0558, -0.6875, -1.6544, -0.6226,  0.1018,  0.0502, -1.2538, 0.1491]]])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> flow.nn.functional.adaptive_avg_pool1d(input, output_size=[4])
        tensor([[[-0.3158, -1.1385,  0.0760, -0.5524]]], dtype=oneflow.float32)

    """,
)
add_docstr(
    oneflow._C.adaptive_avg_pool2d,
    """
    adaptive_avg_pool2d(input, output_size) -> Tensor

    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~oneflow.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        input: the input tensor
        output_size: the target output size (single integer or double-integer tuple)

    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([[[[ 0.1004,  0.0488, -1.0515,  0.9466],[ 0.4538,  0.2361,  1.3437,  0.398 ],[ 0.0558, -0.6875, -1.6544, -0.6226],[ 0.1018,  0.0502, -1.2538,  0.1491]]]])
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> outputs = flow.nn.functional.adaptive_avg_pool2d(input, (2, 2))
    """,
)

add_docstr(
    oneflow._C.adaptive_avg_pool3d,
    """
    adaptive_avg_pool3d(input, output_size) -> Tensor

    Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    See :class:`~oneflow.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        input: the input tensor
        output_size: the target output size (single integer or triple-integer tuple)

    For examples:

    .. code-block:: python

        >>> import oneflow as flow         
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 1, 4, 4, 4), dtype=flow.float32)
        >>> output = flow.nn.functional.adaptive_avg_pool3d(input, (2, 2, 2))
    """,
)

add_docstr(
    oneflow._C.avg_pool1d,
    """
    avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor

    Applies a 1D average pooling over an input signal composed of several input planes.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.avg_pool1d.html

    See :class:`~oneflow.nn.AvgPool1d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\\text{minibatch} , \\text{in_channels} , iW)`
        kernel_size: the size of the window. Can be a single number or a tuple `(kW,)`
        stride: the stride of the window. Can be a single number or a tuple `(sW,)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a single number or a tuple `(padW,)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the averaging calculation. Default: ``True``

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> import oneflow
        >>> input = oneflow.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=oneflow.float32)
        >>> oneflow.nn.functional.avg_pool1d(input, kernel_size=3, stride=2)
        tensor([[[2., 4., 6.]]], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow._C.avg_pool2d,
    """
    avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0) -> Tensor

    Applies 2D average-pooling operation in :math:`kH \\times kW` regions by step size :math:`sH \\times sW` steps. The number of output features is equal to the number of input planes.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.avg_pool2d.html.

    See :class:`~oneflow.nn.AvgPool2d` for details and output shape.

    Args:
        input: input tensor :math:`(\\text{minibatch} , \\text{in_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: 0
    """,
)

add_docstr(
    oneflow._C.avg_pool3d,
    """
    avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0) -> Tensor

    Applies 3D average-pooling operation in :math:`kT \\times kH \\times kW` regions by step size :math:`sT \\times sH \\times sW` steps. The number of output features is equal to :math:`\\lfloor\\frac{\\text{input planes}}{sT}\\rfloor`.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.avg_pool3d.html

    See :class:`~oneflow.nn.AvgPool3d` for details and output shape.

    Args:
        input: input tensor :math:`(\\text{minibatch} , \\text{in_channels} , iT \\times iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a single number or a tuple `(padT, padH, padW)`, Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: 0
    """,
)

add_docstr(
    oneflow._C.max_unpool1d,
    """
    max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None) -> Tensor

    Computes a partial inverse of ``MaxPool1d``.

    See :class:`MaxUnpool1d` for details.
    """,
)

add_docstr(
    oneflow._C.max_unpool2d,
    """
    max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None) -> Tensor

    Computes a partial inverse of ``MaxPool2d``.

    See :class:`MaxUnpool2d` for details.
    """,
)

add_docstr(
    oneflow._C.max_unpool3d,
    """
    max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None) -> Tensor

    Computes a partial inverse of ``MaxPool3d``.

    See :class:`MaxUnpool3d` for details.
    """,
)
