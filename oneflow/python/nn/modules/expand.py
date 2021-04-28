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
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional


class Expand(Module):
    """This operator expand the input tensor to a larger size.
    
    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions and the new ones will be appended at the front. 
    
    For the new dimensions, the size cannot be set to -1. 

    Args:
        x (oneflow.Tensor): The input Tensor. 
        expand_size (Sequence[int]): The desired expanded size.

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

        out = flow.tmp.expand(input, expand_size=[1, 3, 2, 2])

        # out shape: [1, 3, 2, 2]
        # [[[[0, 1],
        #    [0, 1]],
        #   [[2, 3],
        #    [2, 3]],
        #   [[4, 5],
        #    [4, 5]]]]
    """

    def __init__(self, expand_size) -> None:
        super().__init__()
        self._op = flow.builtin_op("expand").Input("in").Output("out").Build()
        self.expand_size = expand_size

    def forward(self, x):
        expand_size = list(self.expand_size)
        assert len(expand_size) >= len(
            x.shape
        ), "The desired expanded dims should not be less than the input dims."
        # calculate the original stride
        original_stride = [1]
        for i in range(len(x.shape) - 2, -1, -1):
            original_stride.insert(0, original_stride[0] * x.shape[i + 1])

        # calculate the output shape and stride
        new_size = []
        new_stride = []
        diff = len(expand_size) - len(x.shape)
        for i in range(len(expand_size) - 1, -1, -1):
            if i >= diff:
                if expand_size[i] == -1 or expand_size[i] == x.shape[i - diff]:
                    new_size.insert(0, x.shape[i - diff])
                    new_stride.insert(0, original_stride[i - diff])
                else:
                    assert expand_size[i] >= 1 and x.shape[i - diff] == 1
                    new_size.insert(0, expand_size[i])
                    new_stride.insert(0, 0)
            else:
                assert expand_size[i] >= 1
                new_size.insert(0, expand_size[i])
                if expand_size[i] == 1:
                    new_stride.insert(0, new_stride[0])
                else:
                    new_stride.insert(0, 0)

        return self._op(
            x, in_shape=list(x.shape), out_shape=new_size, stride=new_stride
        )[0]


@oneflow_export("tmp.expand")
@register_tensor_op("expand")
def expand_op(tensor, expand_size):
    return Expand(expand_size=expand_size)(tensor)
