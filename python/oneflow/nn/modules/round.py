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


def round_op(input):
    """This operator rounds the value of Blob to the nearest integer.
    Args:
        input (oneflow.Tensor): A Tensor
    Returns:
        oneflow.Tensor: The result Tensor
    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([1.49999, 1.500001, 2.7]).astype(np.float32))
        >>> out1 = flow.round(x1)
        >>> out1.numpy()
        array([1., 2., 3.], dtype=float32)
        >>> x2 = flow.Tensor(np.array([2.499999, 7.5000001, 5.3, 6.8]).astype(np.float32))
        >>> out2 = flow.round(x2)
        >>> out2.numpy()
        array([2., 8., 5., 7.], dtype=float32)

    """
    return flow.F.round(input)


@register_tensor_op("round")
def round_op_tensor(input):
    """
    round() -> Tensor

    See :func:`oneflow.round`

    """
    return flow.F.round(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
