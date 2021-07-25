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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Sinh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.sinh(x)


def sinh_op(x):
    """Returns a new tensor with the hyperbolic sine of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\sinh(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow

        >>> x1 = flow.Tensor(np.array([1, 2, 3]))
        >>> x2 = flow.Tensor(np.array([1.53123589,0.54242598,0.15117185]))
        >>> x3 = flow.Tensor(np.array([1,0,-1]))

        >>> flow.enable_eager_execution()
        >>> flow.sinh(x1).numpy()
        array([ 1.1752012,  3.6268604, 10.017875 ], dtype=float32)
        >>> flow.sinh(x2).numpy()
        array([2.20381  , 0.5694193, 0.1517483], dtype=float32)
        >>> flow.sinh(x3).numpy()
        array([ 1.1752012,  0.       , -1.1752012], dtype=float32)

    """
    return Sinh()(x)


@register_tensor_op("sinh")
def sinh_op_tensor(x):
    """

    sinh() -> Tensor

    See :func:`oneflow.compatible.single_client.experimental.sinh`

    """
    return Sinh()(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
