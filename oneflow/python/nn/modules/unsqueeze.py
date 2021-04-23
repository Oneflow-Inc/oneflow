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


@oneflow_export("nn.Unsqueeze")
@register_tensor_op_by_module("unsqueeze")
@register_op_by_module("unsqueeze")
class Unsqueeze(Module):
    r"""Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range ``[0, input.dim() + 1)``
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.dim() + 1``.

    Args:
        input (Tensor) – the input tensor.
        dim (int): the index at which to insert the singleton dimension
    
    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow

        x = flow.Tensor(np.random.rand(2, 3, 4))
        y = x.unsqueeze(2)
        # y.shape >> (2, 3, 1, 4)
    
    """

    def __init__(self, dim: int = 0,) -> None:
        super().__init__()
        self.dim = dim
        self._op = flow.builtin_op("expand_dims").Input("in").Output("out").Build()

    def forward(self, input):
        assert (
            0 <= self.dim <= len(input.size())
        ), "dim should large than 0 and less than the size of input tensor!"
        return self._op(input, axis=self.dim)[0]
