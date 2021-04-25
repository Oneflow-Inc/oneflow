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

from typing import Optional, Sequence, Union
import collections
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util
from oneflow.python.framework.tensor import (
    register_tensor_op_by_module,
    register_op_by_module,
)


@register_op_by_module("pow")
@oneflow_export("Pow")
class Pow(Module):
    r"""Takes the power of each element in input with exponent and returns a tensor with the result.
    exponent can be either a single float number or a single int number.
    
    For example:
    .. code-block:: python
        # Example
        pow = flow.Pow()
        x = flow.Tensor(np.array([1,2,3,4,5,6]))
        out = pow(x,2).numpy()
        print(out) # [1,4,9,16,25,36]
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_pow", name).Input("in").Output("out").Build()

    def forward(self, x, exponent: Union[int, float]):
        return self._op(x, exponent=float(exponent))[0]
