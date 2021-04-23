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
import numpy as np
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module


@oneflow_export("MaskedFill")
@register_op_by_module("tmp.masked_fill")
@register_tensor_op_by_module("masked_fill")
class MaskedFill(Module):
    r"""
    Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is True. 
    The shape of :attr:`mask` must be broadcastable with the shape of the underlying tensor.

    Args:
        mask (BoolTensor) – the boolean mask
        value (float) – the value to fill in with

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        in_arr = np.array(
            [[[-0.13169311,  0.97277078,  1.23305363,  1.56752789],
            [-1.51954275,  1.87629473, -0.53301206,  0.53006478],
            [-1.38244183, -2.63448052,  1.30845795, -0.67144869]],

            [[ 0.41502161,  0.14452418,  0.38968   , -1.76905653],
            [ 0.34675095, -0.7050969 , -0.7647731 , -0.73233418],
            [-1.90089858,  0.01262963,  0.74693893,  0.57132389]]]
        )

        fill_value = 8.7654321 # random value e.g. -1e9 3.1415
        input = flow.Tensor(in_arr, dtype=flow.float32)
        mask = flow.Tensor((in_arr > 0).astype(np.int8), dtype=flow.int)

        output = input.masked_fill(mask, fill_value) 
        #  [[[-0.13169311  8.765432    8.765432    8.765432  ]
        #   [-1.5195427   8.765432   -0.53301203  8.765432  ]
        #   [-1.3824419  -2.6344805   8.765432   -0.6714487 ]]

        #  [[ 8.765432    8.765432    8.765432   -1.7690566 ]
        #   [ 8.765432   -0.7050969  -0.7647731  -0.7323342 ]
        #   [-1.9008986   8.765432    8.765432    8.765432  ]]]

    """

    def __init__(self) -> None:
        super().__init__()
        self._where_op = flow.builtin_op("where").Input("condition").Input("x").Input("y").Output("out").Build()

    def forward(self, input, mask, value):
        in_shape = tuple(input.shape)
        value_like_x = flow.Tensor(*in_shape)
        value_like_x.fill_(value)
        return self._where_op(mask, value_like_x, input)[0]


