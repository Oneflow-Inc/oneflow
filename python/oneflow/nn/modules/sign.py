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


def sign_op(input):
    """Computes the sign of Tensor.

    .. math::

        \\text{out}_{i}  = \\text{sgn}(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([-2, 0, 2]).astype(np.float32))
        >>> out1 = flow.sign(x1)
        >>> out1.numpy()
        array([-1.,  0.,  1.], dtype=float32)
        >>> x2 = flow.Tensor(np.array([-3.2, -4.5, 5.8]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.sign(x2)
        >>> out2.numpy()
        array([-1., -1.,  1.], dtype=float32)

    """
    return flow.F.sign(input)


@register_tensor_op("sign")
def sign_op_tensor(input):
    """

    sign() -> Tensor

    See :func:`oneflow.sign`

    """
    return flow.F.sign(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
