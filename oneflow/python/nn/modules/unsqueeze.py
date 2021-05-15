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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


class Unsqueeze(Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self._op = flow.builtin_op("expand_dims").Input("in").Output("out").Build()

    def forward(self, input):
        assert (
            -(1 + input.ndimension()) <= self.dim <= input.ndimension()
        ), "dim should within the range [-input.ndimension() - 1, input.ndimension() + 1)"

        if self.dim < 0:
            self.dim = 1 + input.ndimension() + self.dim
        return self._op(input, axis=self.dim)[0]


@oneflow_export("unsqueeze")
@register_tensor_op("unsqueeze")
@experimental_api
def unsqueeze_op(input, dim):
    r"""Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range `[-input.ndimension() - 1, input.ndimension() + 1)`
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.ndimension() + 1``.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow

        x = flow.Tensor(np.random.rand(2, 3, 4))
        y = x.unsqueeze(2)

    """
    return Unsqueeze(dim)(input)
