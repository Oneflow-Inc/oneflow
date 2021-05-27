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
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp
        @flow.global_function()
        def acosh_Job(x: tp.Numpy.Placeholder((3,))
        ) -> tp.Numpy:
            return flow.math.acosh(x)
        x = np.array([1.5, 2.5, 3.5]).astype(np.float32)
        out = acosh_Job(x)
        # out [0.96242365 1.56679924  1.9248473 ]
    """

    return Acosh()(x)
