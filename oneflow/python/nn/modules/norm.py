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
import pdb
class Norm(Module):
    def __init__(self, ord = None, dim = None, keepdim = False) -> None:
        super().__init__()

        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim


    def _vector_norm(self, x, ord):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            raise ValueError("Norm order {} is not supported for vectors".format(ord))
        elif isinstance(ord, float) and ord in [float('inf'),float('-inf')]:
            raise NotImplementedError
        elif isinstance(ord, int):
            if ord == 0:
                # TODO: fix error when input are all zero vector
                return flow.tensor([flow.experimental.argwhere(x).shape[0]])
            else:
                return flow.experimental.pow(flow.experimental.sum(flow.experimental.pow(flow.experimental.abs(x), ord)), 1./ord)
        else:
            raise ValueError("Invalid norm order: {}".format(ord))
        
    
    def _matrix_norm(self, x, ord, dim):
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

    def _which_norm(self, x, ord, num_axes, dim):
        if num_axes == 1:
            if ord == None:
                return self._vector_norm(x, ord = 2)
            else:
                return self._vector_norm(x, ord)
        else:
            if ord == None:
                return self._matrix_norm(x, ord = "fro", dim = dim)
            else:
                return self._matrix_norm(x, ord, dim = dim)
    
    def _whether_keepdim(self, x, axis, num_axis):
        if self.keepdim == True and num_axis > 1:
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
            res = self._which_norm(x,self.ord, num_axes, self.dim)
        elif self.dim == None:
            axis = -1
            res = self._which_norm(x, self.ord, num_axes,  self.dim)
        else:
            raise ValueError("Invalid dimension: {}".format(self.dim))
        return self._whether_keepdim(res, axis, num_axes)

        


@oneflow_export("norm")
@register_tensor_op("norm")
@experimental_api
def norm_op(input, ord = None, dim = None, keepdim = False):
    r"""

    """
    return Norm(ord, dim, keepdim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
