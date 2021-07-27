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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.module import Module


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.

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
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import oneflow.compatible.single_client.experimental.nn as nn
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
        self.output_size = output_size
        self._op = (
            flow.builtin_op("adaptive_avg_pool2d")
            .Input("x")
            .Attr("output_size", [])
            .Output("y")
            .Build()
        )

    def forward(self, x):
        new_output_size = []
        assert len(x.shape) == 4
        if isinstance(self.output_size, int):
            new_output_size.append(self.output_size)
            new_output_size.append(self.output_size)
        elif isinstance(self.output_size, tuple):
            new_output_size = list(self.output_size)
            if self.output_size[0] is None:
                new_output_size[0] = x.shape[2]
            if self.output_size[1] is None:
                new_output_size[1] = x.shape[3]
        else:
            raise NotImplementedError("output_size param wrong, please check!")
        new_output_size = tuple(new_output_size)
        assert (
            new_output_size[0] <= x.shape[2]
        ), f"output_size param wrong, please check!"
        assert (
            new_output_size[1] <= x.shape[3]
        ), f"output_size param wrong, please check!"
        return self._op(x, output_size=new_output_size)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
