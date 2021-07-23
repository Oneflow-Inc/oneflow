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


class Eye(Module):
    def __init__(self, m=None):
        super().__init__()
        self.m = m

    def forward(self, n):
        m = self.m
        if m == None or m == n:
           return flow.F.diag(flow.F.ones(n))
        elif m < n:
            tmp = flow.ones(m)
            input1 = flow.zeros([n-m, m])
            input2 = flow.F.diag(tmp)
            res = flow.cat([input1, input2], dim = 0)
            return res
        else:
            tmp = flow.ones(n)
            input1 = flow.zeros([n, m-n])
            input2 = flow.F.diag(tmp)
            res = flow.cat([input1, input2], dim = 1)
            return res


@oneflow_export("eye")
@experimental_api
def eye_op(n, m=None):
    r"""
    This operator creates a 2-D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): the number of rows
        m (Optional[int], optional): the number of colums with default being n. Defaults to None.
    
    Returns:
        oneflow.Tensor: The result Blob with ones on the diagonal and zeros elsewhere.
    
    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> out = flow.eye(3, 3)

        [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
    
    """
    return Eye(m)(n)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)