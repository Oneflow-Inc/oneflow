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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.common_types import _size_1_t
from oneflow.python.nn.modules.utils import _single, _pair, _triple


def _generate_output_size(input_size, output_size):
    new_output_size = []
    assert len(input_size) - 2 == len(
        output_size
    ), f"the length of 'output_size' does not match the input size, {len(input_size) - 2} expected"
    for i in range(len(output_size)):
        if output_size[i] is None:
            new_output_size.append(input_size[i + 2])
        else:
            assert isinstance(
                output_size[i], int
            ), "numbers in 'output_size' should be integer"
            new_output_size.append(output_size[i])
    return tuple(new_output_size)


@oneflow_export("nn.AdaptiveAvgPool1d")
@experimental_api
class AdaptiveAvgPool1d(Module):
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size H

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> import oneflow.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = flow.Tensor(np.random.randn(1, 64, 8))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 5])

    """

    def __init__(self, output_size: _size_1_t) -> None:
        super().__init__()
        assert output_size is not None
        self.output_size = _single(output_size)

    def forward(self, x):
        assert len(x.shape) == 3
        assert len(self.output_size) == 1,  f"the length of 'output_size' does not match the input size, 1 expected"
        assert isinstance(self.output_size[0], int), "numbers in 'output_size' should be integer"
        return flow.F.adaptive_avg_pool1d(x, output_size=self.output_size)


@oneflow_export("adaptive_avg_pool1d")
@experimental_api
def adaptive_avg_pool1d(input, output_size):
    r"""Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.experimental.nn.AdaptiveAvgPool1d`

    Args:
        input: input tensor
        output_size: the target output size (single integer)
    """
    return AdaptiveAvgPool1d(output_size)(input)


@oneflow_export("nn.AdaptiveAvgPool2d")
@experimental_api
class AdaptiveAvgPool2d(Module):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> import oneflow.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = flow.Tensor(np.random.randn(1, 64, 8, 9))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 5, 7])

        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 7, 7])

        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 10, 7])

    """

    def __init__(self, output_size) -> None:
        super().__init__()
        assert output_size is not None
        self.output_size = _pair(output_size)

    def forward(self, x):
        assert len(x.shape) == 4
        new_output_size = _generate_output_size(x.shape, self.output_size)
        return flow.F.adaptive_avg_pool2d(x, output_size=new_output_size)


@oneflow_export("adaptive_avg_pool2d")
@experimental_api
def adaptive_avg_pool2d(input, output_size):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.experimental.nn.AdaptiveAvgPool2d`

    Args:
        input: input tensor
        output_size: the target output size (single integer or double-integer tuple)
    """
    return AdaptiveAvgPool2d(output_size)(input)


@oneflow_export("nn.AdaptiveAvgPool3d")
@experimental_api
class AdaptiveAvgPool3d(Module):
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> import oneflow.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = flow.Tensor(np.random.randn(1, 64, 8, 9, 10))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 5, 7, 9])

        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9, 8))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 7, 7, 7])

        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9, 8))
        >>> output = m(input)
        >>> output.size()
        flow.Size([1, 64, 7, 9, 8])

    """

    def __init__(self, output_size) -> None:
        super().__init__()
        assert output_size is not None
        self.output_size = _triple(output_size)

    def forward(self, x):
        assert len(x.shape) == 5
        new_output_size = _generate_output_size(x.shape, self.output_size)
        return flow.F.adaptive_avg_pool3d(x, output_size=new_output_size)


@oneflow_export("adaptive_avg_pool3d")
@experimental_api
def adaptive_avg_pool3d(input, output_size):
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.experimental.nn.AdaptiveAvgPool3d`

    Args:
        input: input tensor
        output_size: the target output size (single integer or triple-integer tuple)
    """
    return AdaptiveAvgPool3d(output_size)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
