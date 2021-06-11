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

class Norm(Module):
    def __init__(self, ord = None, dim = None, keepdim = False) -> None:
        super().__init__()

        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def _vector_norm(self, x, ord = 2):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            raise ValueError("Norm order {} is not supported for vectors".format(ord))
        elif isinstance(ord, float) and ord in [float('inf'),float('-inf')]:
            raise NotImplementedError
        elif isinstance(ord, int):
            if ord == 0:
                return flow.argwhere(x).shape
            else:
                return flow.experimental.pow(flow.experimental.sum(flow.experimental.pow(flow.experimental.abs(x), ord)), 1./ord)
        else:
            raise ValueError("Invalid norm order: {}".format(ord))
        
    
    def _matrix_norm(self, x, ord = "fro", dim = None):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            if ord == "nuc":
                raise NotImplementedError
            else:
                return flow.experimental.sqrt(flow.experimental.sum(flow.experimental.square(x), dim = dim))
        elif isinstance(ord, float) and ord in [float('inf'),float('-inf')]:
            raise NotImplementedError
        elif isinstance(ord, int):
            if ord == 1:
                raise NotImplementedError
            elif ord == -1:
                raise NotImplementedError
            elif ord == 2:
                raise NotImplementedError
            elif ord == -2:
                raise NotImplementedError
            else:
                raise ValueError("Norm order {} is not supported for matrices".format(ord))
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def _which_norm(self, x, num_axes, ord, dim = None):
        if num_axes == 1:
            return self._vector_norm(x, ord)
        else:
            return self._matrix_norm(x, ord, dim)

    def _whether_keepdim(self, x, axis):
        if self.keepdim == True and axis:
            return flow.experimental.unsqueeze(x, axis)
        else:
            return x

    def forward(self, x):
        num_axes = len(x.shape)
        if isinstance(self.dim, (int,tuple)):
            if isinstance(self.dim, int):
                axis = self.dim if self.dim >= 0 else self.dim + num_axes
                assert 0 <= axis < num_axes, "axis out of range"
            else:
                raise NotImplementedError
            res = self._which_norm(x, num_axes, self.ord, self.dim)
        elif self.dim == None:
            axis = -1
            res = self._which_norm(x, num_axes, self.ord, self.dim)
        else:
            raise ValueError("Invalid dimension: {}".format(self.dim))
        

        


# @oneflow_export("norm")
# @register_tensor_op("norm")
# @experimental_api
# def norm_op(input, ord = None, dim = None, keepdim = False):
#     r"""Returns a new tensor with the natural logarithm of (1 + input).

#     .. math::
#         \text{out}_{i}=\log_e(1+\text{input}_{i})

#     For example:

#     .. code-block:: python

#         >>> import oneflow.experimental as flow
#         >>> import numpy as np
#         >>> flow.enable_eager_execution()
#         >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
#         >>> out = flow.log1p(x).numpy()
#         >>> out
#         array([0.8329091 , 0.91629076, 1.3083328 ], dtype=float32)

#     """
#     return Norm(ord = None, dim = None, keepdim = False)(input)


if __name__ == "__main__":
    import numpy as np
    flow.experimental.enable_eager_execution()
    np_a = np.arange(8).astype(np.float32) - 4
    np_b = np_a.reshape(2,2,2)
    f_a = flow.experimental.tensor(np_a,dtype=flow.float32)
    f_b = flow.experimental.tensor(np_b,dtype=flow.float32)
    # print(f_a,f_b)
    # x = Norm(ord = "fro",keepdim=True)(f_b)
    # x = Norm(ord = 2)(f_a)
    print(flow.experimental.argwhere(f_a).shape)
    # import doctest

    # doctest.testmod(raise_on_error=True)
