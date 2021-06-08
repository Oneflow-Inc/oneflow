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
from __future__ import absolute_import

from typing import Union

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


@oneflow_export("nn.ConstantPad2d")
@experimental_api
class ConstantPad2d(Module):
    r"""This operator pads the input with constant value that user specifies. User can set the amount of padding by setting the parameter `paddings`.

    Args:
        padding (Union[int, tuple, list]):  the size of the padding. If is `int`, uses the same padding in all boundaries. If a 4-`tuple`, uses (:math:`\mathrm{padding_{left}}`, :math:`\mathrm{padding_{right}}`, :math:`\mathrm{padding_{top}}`, :math:`\mathrm{padding_{bottom}}`)
        
        value (Union[int, float]): The constant value used for padding. Defaults to 0.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

            :math:`H_{out} = H_{in} + \mathrm{padding_{top}} + \mathrm{padding_{bottom}}`

            :math:`W_{out} = W_{in} + \mathrm{padding_{left}} + \mathrm{padding_{right}}`

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> constantpad_layer_0 = flow.nn.ConstantPad2d((2, 2, 1, 1), 1)
        >>> input = flow.Tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = constantpad_layer_0(input)
        >>> print(output.shape)
        flow.Size([1, 2, 5, 7])
        >>> print(output.numpy())
        [[[[ 1.  1.  1.  1.  1.  1.  1.]
           [ 1.  1.  0.  1.  2.  1.  1.]
           [ 1.  1.  3.  4.  5.  1.  1.]
           [ 1.  1.  6.  7.  8.  1.  1.]
           [ 1.  1.  1.  1.  1.  1.  1.]]
        <BLANKLINE>
          [[ 1.  1.  1.  1.  1.  1.  1.]
           [ 1.  1.  9. 10. 11.  1.  1.]
           [ 1.  1. 12. 13. 14.  1.  1.]
           [ 1.  1. 15. 16. 17.  1.  1.]
           [ 1.  1.  1.  1.  1.  1.  1.]]]]
        >>> constantpad_layer_1 = flow.nn.ConstantPad2d((1, 2, 3, 4), 0.5)
        >>> output_1 = constantpad_layer_1(input)
        Padding shape must be less than input shape. Please check.
    """

    def __init__(
        self,
        padding: Union[int, tuple, list],
        value: Union[int, float] = 0
    ):
        super().__init__()
        if isinstance(padding, (tuple, list)):
            assert len(padding) == 4, ValueError("Padding length must be 4")
            boundary = [padding[0], padding[1], padding[2], padding[3]]
        elif isinstance(padding, int):
            boundary = [padding, padding, padding, padding]
        else:
            raise ValueError("padding must be in or list or tuple!")

        self.padding = boundary
        self._op = (
            flow.builtin_op("constant_pad2d")
            .Input("x")
            .Output("y")
            .Attr("padding", self.padding)
            .Attr("floating_value", float(value))
            .Attr("integral_value", int(0))
            .Build()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if self.padding[2] < h and self.padding[3] < h and self.padding[0] < w and self.padding[1] < w:
            res = self._op(x)[0]
            return res
        else:
            print("Padding shape must be less than input shape. Please check.")
            return

if __name__ == "__main__":
    import doctest
    doctest.testmod()
