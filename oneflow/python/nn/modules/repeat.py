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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module
from oneflow.python.ops.transpose_util import (
    get_perm_when_transpose_axis_to_last_dim,
    get_inversed_perm,
)


@oneflow_export("Repeat")
@register_tensor_op_by_module("tmp.repeat")
@register_op_by_module("tmp.repeat")
class Repeat(Module):
    """This operator repeat the input tensor to a larger size along the specified dimensions.

    Args:
        x (oneflow.Tensor): The input Tensor. 
        size (Sequence[int]): The number of times to repeat this tensor along each dimension

    Returns:
        oneflow.Tensor: The result Tensor. 

    For example: 

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        x = np.array([[[[0, 1]],
                       [[2, 3]],
                       [[4, 5]]]]).astype(np.int32)

        input = flow.Tensor(x)

        out = flow.tmp.repeat(input, sizes=[1, 1, 2, 2]).numpy()

        # out shape: [1, 3, 2, 4]
        # [[[[0. 1. 0. 1.]
        # [0. 1. 0. 1.]]

        # [[2. 3. 2. 3.]
        # [2. 3. 2. 3.]]

        # [[4. 5. 4. 5.]
        # [4. 5. 4. 5.]]]]
    """

    def __init__(self, sizes) -> None:
        super().__init__()
        self.sizes = sizes

    def forward(self, input):
        repeat = self.sizes
        input_shape = input.shape
        assert len(repeat) >= len(input_shape)
        in_reshape = []
        out_reshape = []
        expand_dim = []
        diff = len(repeat) - len(input_shape)
        for i in range(len(repeat) - 1, -1, -1):
            if i >= diff:
                if repeat[i] > 1:
                    if input_shape[i - diff] > 1:
                        in_reshape.insert(0, input_shape[i - diff])
                        in_reshape.insert(0, 1)
                        expand_dim.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, input_shape[i - diff] * repeat[i])
                    else:
                        in_reshape.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, repeat[i])
                else:  # repeat[i] == 1
                    in_reshape.insert(0, input_shape[i - diff])
                    expand_dim.insert(0, input_shape[i - diff])
                    out_reshape.insert(0, input_shape[i - diff])
            else:  # i < diff
                expand_dim.insert(0, repeat[i])
                out_reshape.insert(0, repeat[i])

        new_tensor = flow.tmp.reshape(input, in_reshape)
        tmp_tensor = flow.tmp.expand(new_tensor, expand_dim)
        out = flow.tmp.reshape(tmp_tensor, out_reshape)
        return out
