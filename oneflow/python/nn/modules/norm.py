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
        if ord in ["fro", "nuc"]:
            raise ValueError("Norm order {} is not supported for vectors".format(ord))
        elif ord in [float('inf'),float('-inf')]:
            return self._max_op(self._abs_op(x)[0])[0]
        elif ord.isdigit():
            if ord == 0:
                raise NotImplementedError
            else:
                return flow.experimental.pow(flow.experimental.sum(flow.experimental.pow(flow.experimental.abs(x)[0], ord)), 1./ord)
        else:
            raise ValueError("Invalid norm order: {}".format(ord))
        
    
    def _matrix_norm(self, x, ord = "fro", dim = None):
        if ord in ["fro", "nuc"]:
            assert len(x.shape) == 2, "Both the Frobenius and nuclear norm orders are only defined for matrices"
            if ord == "nuc":
                raise NotImplementedError
            else:
                return flow.experimental.sqrt(flow.experimental.sum(flow.experimental.square(x), dim = dim))
        elif ord in [float('inf'),float('-inf')]:
            return self._max_op((flow.experimental.abs(x)))
        elif ord.isdigit():
            if ord == 1:
                return
            elif ord == -1:
                return 
            elif ord == 2:
                raise NotImplementedError
            elif ord == -2:
                raise NotImplementedError
            else:
                raise ValueError("Norm order {} is not supported for matrices".format(ord))
        else:
            raise ValueError("Invalid norm order: {}".format(ord))




    def forward(self, x):
        num_axes = len(x.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if num_axes == 1:
            res = self._vector_norm(x, self.ord)
        else:
            res = self._matrix_norm(x, self.dim)

        if axis == num_axes - 1:
            if self.keepdim == True:
                res = flow.experimental.unsqueeze(res, -1)
            return res
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = self._transpose_op(input, perm=perm)[0]
            x = self._op_softmax_last_dim(x)[0]
            x = flow.experimental.unsqueeze(x, -1)
            x = self._transpose_op(x, perm=get_inversed_perm(perm))[0]
            if self.keepdim == False:
                x = x.squeeze(dim=[axis])
            return x
        


@oneflow_export("norm")
@register_tensor_op("norm")
@experimental_api
def norm_op(input, ord = None, dim = None, keepdim = False):
    r"""Returns a new tensor with the natural logarithm of (1 + input).

    .. math::
        \text{out}_{i}=\log_e(1+\text{input}_{i})

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> x = flow.Tensor(np.array([1.3, 1.5, 2.7]))
        >>> out = flow.log1p(x).numpy()
        >>> out
        array([0.8329091 , 0.91629076, 1.3083328 ], dtype=float32)

    """
    return Norm(ord = None, dim = None, keepdim = False)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
