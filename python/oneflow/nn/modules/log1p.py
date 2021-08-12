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


def log1p_op(input):
    """Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \\text{out}_{i}=\\log_e(1+\\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log1p(x)
        >>> out
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

    """
    return flow.F.log1p(input)


@register_tensor_op("log1p")
def log1p_op_tensor(input):
    """
    See :func:`oneflow.log1p`
    """
    return flow.F.log1p(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
