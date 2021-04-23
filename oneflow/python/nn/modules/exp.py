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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op_by_module
from oneflow.python.framework.tensor import register_op_by_module


@oneflow_export("Exp")
@register_tensor_op_by_module("exp")
@register_op_by_module("exp")
class Exp(Module):
    r"""
    Returns a new tensor with the exp of the elements of :attr:`x`.
    .. math::
        \text{y}_{i} = \exp(\text{x}_{i})
    Args:
        {x}
    
    For example: 
    .. code-block:: python 
        import numpy as np
        import oneflow as flow
        x = flow.Tensor(np.random.rand(2, 3, 4))
        y = x.exp()
        print(y.shape)
        # (2, 3, 4)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("exp").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]
