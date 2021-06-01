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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module


class Acosh(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("acosh").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("acosh")
@register_tensor_op("acosh")
@experimental_api
def acosh_op(x):
    r"""Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.
    
    .. math::

        \text{out}_{i} = \acosh^{-1}(\text{input}_{i})
    Args:
        input (Tensor): the input tensor.
    For example:
    .. code-block:: python
        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> import oneflow.typing as tp
        >>> flow.enable_eager_execution()
        >>> x1 = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        >>> out1 = flow.math.acosh(x1)
        >>> out1.numpy()
        [1.316958 1.7627473  2.063437 ]

    """

    return Acosh()(x)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
