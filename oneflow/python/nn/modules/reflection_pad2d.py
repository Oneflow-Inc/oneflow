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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from typing import Union


@oneflow_export("nn.ReflectionPad2d")
@experimental_api
class ReflectionPad2d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from:
    https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html


    This operator pads the input tensor using the reflection of the input boundary.

    Args:
        padding (Union[int,tuple]): The size or bundary of padding, if is `int` uses the same padding in all dimension; if 4-dims `tuple`, uses :math:`(\text{padding}_{\text{left}}, \text{padding}_{\text{right}}, \text{padding}_{\text{top}}, \text{padding}_{\text{bottom}} )`

    Returns:
        Tensor: Returns a new tensor which is result of the reflection padding of the input tensor.

    Shape:
        - Input: :math:`(N, C, H_{\text{in}}, W_{\text{in}})`
        - Output: :math:`(N, C, H_{\text{out}}, W_{\text{out}})` where

          :math:`H_{\text{out}} = H_{\text{in}} + \text{padding}_{\text{top}} + \text{padding}_{\text{bottom}}`

          :math:`W_{\text{out}} = W_{\text{in}} + \text{padding}_{\text{left}} + \text{padding}_{\text{right}}`

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor(np.arange(18).reshape((1, 2, 3, 3)), dtype=flow.float32)
        >>> m = flow.nn.ReflectionPad2d((2, 2, 1, 1))
        >>> out = m(input)
        >>> out
        tensor([[[[ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
                  [ 2.,  1.,  0.,  1.,  2.,  1.,  0.],
                  [ 5.,  4.,  3.,  4.,  5.,  4.,  3.],
                  [ 8.,  7.,  6.,  7.,  8.,  7.,  6.],
                  [ 5.,  4.,  3.,  4.,  5.,  4.,  3.]],
        <BLANKLINE>         
                 [[14., 13., 12., 13., 14., 13., 12.],
                  [11., 10.,  9., 10., 11., 10.,  9.],
                  [14., 13., 12., 13., 14., 13., 12.],
                  [17., 16., 15., 16., 17., 16., 15.],
                  [14., 13., 12., 13., 14., 13., 12.]]]], dtype=oneflow.float32)

    """

    def __init__(self, padding: Union[int, tuple]) -> None:
        super().__init__()
        if isinstance(padding, tuple):
            assert len(padding) == 4, ValueError("Padding length must be 4")
            boundary = [padding[0], padding[1], padding[2], padding[3]]
        elif isinstance(padding, int):
            boundary = [padding, padding, padding, padding]
        else:
            raise ValueError("padding must be in or list or tuple!")
        self.padding = boundary
        self._op = (
            flow.builtin_op("reflection_pad2d")
            .Input("x")
            .Output("y")
            .Attr("padding", boundary)
            .Attr("floating_value", float(1.0))
            .Attr("integral_value", int(0))
            .Build()
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        if (
            self.padding[2] < H
            and self.padding[3] < H
            and self.padding[0] < W
            and self.padding[1] < W
        ):
            res = self._op(x)[0]
            return res
        else:
            raise ValueError(
                "padding size should be less than the corresponding input dimension!"
            )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
