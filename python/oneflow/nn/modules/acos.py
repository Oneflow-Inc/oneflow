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


@register_tensor_op("acos")
def acos_op(input):
    """
    Returns a new tensor with the inverse cosine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\arccos(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> arr = np.array([0.5, 0.6, 0.7])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)

    """
    return flow.F.acos(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
