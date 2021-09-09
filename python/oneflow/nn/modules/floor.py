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


def floor_op(input):
    """
    Returns a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\lfloor \\text{input}_{i} \\rfloor

    Args:
        input (Tensor): the input tensor.
        
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([-0.5,  1.5, 0,  0.8]), dtype=flow.float32)
        >>> output = flow.floor(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output.numpy()
        array([-1.,  1.,  0.,  0.], dtype=float32)
        
        >>> input1 = flow.tensor(np.array([[0.8, 1.0], [-0.6, 2.5]]), dtype=flow.float32)
        >>> output1 = input1.floor()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1.numpy()
        array([[ 0.,  1.],
               [-1.,  2.]], dtype=float32)

    """
    return flow._C.floor(input)


@register_tensor_op("floor")
def floor_op_tensor(input):
    """
    See :func:`oneflow.floor`
    """
    return flow._C.floor(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
