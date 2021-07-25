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
import collections
from typing import Optional, Sequence, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _check_axis


class Floor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.floor(x)


def floor_op(x):
    """
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\lfloor \\text{input}_{i} \\rfloor

    Args:
        input (Tensor): the input tensor.
        
    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> input = flow.Tensor(np.array([-0.5,  1.5, 0,  0.8]), dtype=flow.float32)
        >>> output = flow.floor(input)
        >>> output.shape
        flow.Size([4])
        >>> output.numpy()
        array([-1.,  1.,  0.,  0.], dtype=float32)
        
        >>> input1 = flow.Tensor(np.array([[0.8, 1.0], [-0.6, 2.5]]), dtype=flow.float32)
        >>> output1 = input1.floor()
        >>> output1.shape
        flow.Size([2, 2])
        >>> output1.numpy()
        array([[ 0.,  1.],
               [-1.,  2.]], dtype=float32)

    """
    return Floor()(x)


@register_tensor_op("floor")
def floor_op_tensor(input):
    """
    See :func:`oneflow.compatible.single_client.experimental.floor`
    """
    return Floor()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
