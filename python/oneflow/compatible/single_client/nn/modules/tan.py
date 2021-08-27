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


class Tan(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("tan").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


def tan_op(input):
    """Returns  the tan value of the elements of :attr:`input`.

    .. math::
        \\text{out}_{i} = \\tan(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> np_arr = np.array([-1/4*np.pi, 0, 1/4*np.pi]).astype(np.float32)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.tan(input)
        >>> output
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)

    """
    return Tan()(input)


@register_tensor_op("tan")
def tan_op_tensor(input):
    """
    tan() -> Tensor
    See :func:`oneflow.compatible.single_client.experimental.tan`

    """
    return Tan()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
