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
from typing import Tuple, Union

import oneflow as flow
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import experimental_api, oneflow_export


@oneflow_export("nn.ReflectionPad2d")
@experimental_api
class ReflectionPad2d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html
    
    Pads the input tensor using the reflection of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` . only support "NCHW" format.
        - Output: :math:`(N, C, H_{out}, W_{out})` where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> m = flow.nn.ReflectionPad2d(2)
        >>> input = np.arange(9).reshape((1, 1, 3, 3)).astype(np.float32)
        >>> input
        array([[[[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]]]], dtype=float32)
        >>> input = flow.Tensor(input, dtype=flow.float32)
        >>> m(input)
        tensor([[[[8., 7., 6., 7., 8., 7., 6.],
                [5., 4., 3., 4., 5., 4., 3.],
                [2., 1., 0., 1., 2., 1., 0.],
                [5., 4., 3., 4., 5., 4., 3.],
                [8., 7., 6., 7., 8., 7., 6.],
                [5., 4., 3., 4., 5., 4., 3.],
                [2., 1., 0., 1., 2., 1., 0.]]]], dtype=oneflow.float32)
        >>> # using different paddings for different sides
        >>> m = flow.nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[7., 6., 7., 8., 7.],
                [4., 3., 4., 5., 4.],
                [1., 0., 1., 2., 1.],
                [4., 3., 4., 5., 4.],
                [7., 6., 7., 8., 7.]]]], dtype=oneflow.float32)

    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]) -> None:
        super().__init__()

        self.boundry = padding
        if isinstance(padding, int):
            self.boundry = [padding, padding, padding, padding]

        self._op = (
            flow.builtin_op("reflection_pad2d")
            .Input("x")
            .Output("y")
            .Attr("padding", list(self.boundry))
            .Build()
        )

    def forward(self, input: Tensor) -> Tensor:
        H, W = input.shape[2], input.shape[3]
        if isinstance(self.boundry, (tuple, list)):
            assert len(self.boundry) == len(input.shape), ValueError(
                "padding boundry must be the same size of input dims"
            )
            assert (
                self.boundry[2] < H
                and self.boundry[3] < H
                and self.boundry[0] < W
                and self.boundry[1] < W
            ), ValueError(
                "Padding size should be less than the corresponding input dimension!"
            )
        else:
            raise NotImplementedError("padding must be in or list or tuple!")

        return self._op(input)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
