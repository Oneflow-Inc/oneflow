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
