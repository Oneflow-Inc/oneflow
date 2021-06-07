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
    def __init__(
        self,
        padding: Union[int, tuple, list],
        value: Union[int, float] = 0
    ):
        """This operator pads the input blob with constant value that user specifies. User can set the amount of padding by setting the parameter `paddings`.

        Args:
            padding (Union[int, tuple, list]): The size or bundary of padding, if is `int` uses the same padding in all dimension; if 4-dims `tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            value (Union[int, float]): The constant value used for padding. Defaults to 0.

        Shape:
            - Input: :math:`(N, C, H_{in}, W_{in})`
            - Output: :math:`(N, C, H_{out}, W_{out})` where

                :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}` 

                :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}` 

    For example::

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np

        >>> flow.enable_eager_execution()
        >>> constantpad_layer_0 = flow.nn.ConstantPad2d((2, 2, 1, 1), 1)
        >>> input = flow.Tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = constantpad_layer_0(input)
        >>> print(output.shape)
        torch.Size([1, 2, 5, 7])
        >>> print(output.numpy())
        array([[[[ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
         [ 1.,  1.,  0.,  1.,  2.,  1.,  1.],
         [ 1.,  1.,  3.,  4.,  5.,  1.,  1.],
         [ 1.,  1.,  6.,  7.,  8.,  1.,  1.],
         [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]],

        [[ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
         [ 1.,  1.,  9., 10., 11.,  1.,  1.],
         [ 1.,  1., 12., 13., 14.,  1.,  1.],
         [ 1.,  1., 15., 16., 17.,  1.,  1.],
         [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]]]], dtype=float32)

        >>> constantpad_layer_1 = flow.nn.ConstantPad2d((1, 2, 3, 4), 0.5)
        >>> output_1 = constantpad_layer1(input)
        >>> print(output_1.shape)
        torch.Size([1, 2, 10, 6])
        >>> print(output_1.numpy())
        array([[[[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0. ,  1. ,  2. ,  0.5,  0.5],
         [ 0.5,  3. ,  4. ,  5. ,  0.5,  0.5],
         [ 0.5,  6. ,  7. ,  8. ,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5]],

        [[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  9. , 10. , 11. ,  0.5,  0.5],
         [ 0.5, 12. , 13. , 14. ,  0.5,  0.5],
         [ 0.5, 15. , 16. , 17. ,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
         [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5]]]], dtype=float32)
       
    """
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
            .Input("in")
            .Output("out")
            .Attr("padding", self.padding)
            .Attr("floating_value", value)
            .Attr("integral_value", value)
            .Build()
        )

    def forward(self, x):
        res = self._op(x)[0]
        return res

if __name__ == "__main__":
    import doctest
    doctest.testmod()
