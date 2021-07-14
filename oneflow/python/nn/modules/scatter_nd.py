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

from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


@oneflow_export("scatter_nd")
@experimental_api
class Scatter_nd(Module):
    r"""This operator inserts the elements in `updates` according to the `index` and create a new Tensor.

    Args:
        index: The indices of `updates`. Its type should be `flow.int`.
        updates: The update Tensor.
        shape (Sequence[int]): The constant tensor shape, the constant tensor elements are all zero.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> scatter_nd_layer = flow.scatter_nd([8])
        >>> index = flow.Tensor(np.array([[1], [6], [4]]), dtype=flow.int)
        >>> update = flow.Tensor(np.array([10.2,5.1,12.7]), dtype=flow.float)
        >>> out = scatter_nd_layer(index,update)
        >>> out
        tensor([ 0. , 10.2,  0. ,  0. , 12.7,  0. ,  5.1,  0. ], dtype=oneflow.float32)

    """

    def __init__(self, shape: list):
        super().__init__()
        if not isinstance(shape, list):
            raise ValueError("shape must be list!")
        self.shape = shape

    def forward(self, index, updates):
        self._op = (
            flow.builtin_op("scatter_nd")
            .Input("indices")
            .Input("updates")
            .Output("out")
            .Attr("shape", self.shape)
            .Build()
        )

        res = self._op(index, updates)[0]
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
