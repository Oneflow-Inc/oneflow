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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module


class Atan2(Module):
    def __init__(self) -> None:
        super().__init__()
        self.atan2_op = (
            flow.builtin_op("atan2").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self.atan2_op(x, y)[0]


def atan2_op(input, other):
    """Element-wise arctangent of input{i}/other{i}
    with consideration of the quadrant. Returns a new tensor with the signed
    angles in radians between vector (other{i},input{i}) and vector (1, 0).

    The shapes of input and other must be broadcastable.

    Args:
        input (Tensor): the first input tensor.

        other (Tensor): the second input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x1 = flow.Tensor(np.array([1,2,3]))
        >>> y1 = flow.Tensor(np.array([3,2,1]))
        >>> x2 = flow.Tensor(np.array([1.53123589,0.54242598,0.15117185]))
        >>> y2 = flow.Tensor(np.array([-0.21906378,0.09467151,-0.75562878]))
        >>> x3 = flow.Tensor(np.array([1,0,-1]))
        >>> y3 = flow.Tensor(np.array([0,1,0]))

        >>> flow.atan2(x1,y1).numpy()
        array([0.32175055, 0.7853982 , 1.2490457 ], dtype=float32)
        >>> flow.atan2(x2,y2).numpy()
        array([1.7128955, 1.3980033, 2.9441385], dtype=float32)
        >>> flow.atan2(x3,y3).numpy()
        array([ 1.5707964,  0.       , -1.5707964], dtype=float32)

    """
    return Atan2()(input, other)


@register_tensor_op("atan2")
def atan2_op_tensor(input, other):
    """

    atan2(other) -> Tensor

    See :func:`oneflow.atan2`
    """
    return Atan2()(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
