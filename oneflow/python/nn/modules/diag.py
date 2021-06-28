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
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op

class Diag(Module):
    def __init__(self, diagonal):
        super().__init__()
        self._op = flow.builtin_op("diag").Input("in").Output("out").Attr("diagonal", diagonal).Build()

    def forward(self, input):
        return self._op(input)[0]



@oneflow_export("diag")
@register_tensor_op("diag")
@experimental_api
def diag_op(input, diagonal):
    r"""
    Returns a new tensor with the diagonal.

    ..math::

    Args:
        input (Tensor): the input tensor.
        diagonal (Optional[int], 0): The diagonal to consider. If diagonal = 0, it is the main diagonal. If diagonal > 0, it is above the main diagonal. If diagonal < 0, it is below the main diagonal. Defaults to 0.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> arr = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0],])
        >>> input = flow.Tensor(arr, dtype=flow.float32)
        >>> output = flow.diag(input)
        >>> print(output.numpy())
        [1. 5. 9.]
    """

    return Diag(diagonal)(input)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)