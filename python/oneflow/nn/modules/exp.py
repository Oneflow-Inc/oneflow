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


@register_tensor_op("exp")
def exp_op(x):
    """This operator computes the exponential of Tensor.

    The equation is:

    .. math::

        out = e^x

    Args:
        x (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: The result Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = x.exp()
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """
    return flow.F.exp(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
