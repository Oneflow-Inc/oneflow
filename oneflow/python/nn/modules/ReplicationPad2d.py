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

@oneflow_export("nn.ReplicationPad2d")
@experimental_api
class ReplicationPad2d(Module):
    r"""Pads the input tensor using the replication of the input boundary.

    Args:
        padding (Union[int, tuple, list]):  the size of the padding. If is `int`, uses the same padding in all boundaries. If a 4-`tuple`, uses (:math:`\mathrm{padding_{left}}`, :math:`\mathrm{padding_{right}}`, :math:`\mathrm{padding_{top}}`, :math:`\mathrm{padding_{bottom}}`)

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
        >>> replicationpad_layer_0 = flow.nn.ReplicationPad2d((2, 2, 1, 1))
        >>> input = flow.Tensor(np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32))
        >>> output = replicationpad_layer_0(input)
        >>> print(output.shape)
        flow.Size([1, 2, 5, 7])
        >>> print(output.numpy())
        [[[[ 0.  0.  0.  1.  2.  2.  2.]
           [ 0.  0.  0.  1.  2.  2.  2.]
           [ 3.  3.  3.  4.  5.  5.  5.]
           [ 6.  6.  6.  7.  8.  8.  8.]
           [ 6.  6.  6.  7.  8.  8.  8.]]
        <BLANKLINE>
          [[ 9.  9.  9. 10. 11. 11. 11.]
           [ 9.  9.  9. 10. 11. 11. 11.]
           [12. 12. 12. 13. 14. 14. 14.]
           [15. 15. 15. 16. 17. 17. 17.]
           [15. 15. 15. 16. 17. 17. 17.]]]]
        >>> replicationpad_layer_1 = flow.nn.ReplicationPad2d((1, 2, 3, 4))
        >>> output_1 = replicationpad_layer_1(input)
        Padding size should be less than the corresponding input dimension. Please check.

    """
    def __init__(
        self,
        padding: Union[int, tuple, list]
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
            flow.builtin_op("replication_pad2d")
            .Input("x")
            .Output("y")
            .Attr("padding", self.padding)
            .Build()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if self.padding[2] < h and self.padding[3] < h and self.padding[0] < w and self.padding[1] < w:
            res = self._op(x)[0]
            return res
        else:
            print("Padding size should be less than the corresponding input dimension. Please check.")
            return

if __name__ == "__main__":
    import doctest
    doctest.testmod()