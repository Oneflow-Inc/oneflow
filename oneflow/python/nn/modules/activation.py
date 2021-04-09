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
from typing import Optional, Sequence, Sized, Union, List, Tuple
from oneflow.python.nn.modules.array_ops import transpose


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0:
        axis += dim_num
    assert axis >= 0
    assert axis < dim_num

    need_transpose = False
    permute = list(range(dim_num))
    if axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return need_transpose, permute


@oneflow_export("nn.Sigmoid")
@register_tensor_op_by_module("sigmoid")
@register_op_by_module("sigmoid")
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("sigmoid").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.ReLU")
@register_tensor_op_by_module("relu")
@register_op_by_module("relu")
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("relu").Input("in").Output("out").Build()

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.Softmax")
class Softmax(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
    
        m = flow.nn.LogSoftmax(dim=1)
        x = flow.Tensor(
            np.array(
                [[ 0.81733328,  0.43621480,  0.10351428],
                [-1.15555191, -0.67776406,  0.27372134]]
            )
        )
        y = m(x)

        # y:
        # [[0.69367    0.6073567  0.5258555 ]
        # [0.23947646 0.33676052 0.5680063 ]]

    """
    def __init__(
        self, axis: Optional[int] = None, name: Optional[str] = None,
    ):
        super().__init__()

        if axis is None:
            axis = -1
        self.axis = axis

        self._op = flow.builtin_op("softmax", name).Input("in").Output("out").Build()

    def forward(self, x):
        print(x.shape)
        need_transpose, permute = _softmax_need_transpose(x, self.axis)
        if need_transpose:
            logits = flow.transpose(logits, perm=permute)

        res = self._op(x)[0]
        if need_transpose:
            res = transpose(res, perm=permute)
        return res


@oneflow_export("nn.LogSoftmax")
class LogSoftmax(Module):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
    
        m = flow.nn.LogSoftmax(dim=1)
        x = flow.Tensor(
            np.array(
                [[ 0.4296, -1.1957,  2.5463],
                [ 1.2552, -1.5747,  0.6923]]
            )
        )
        y = m(x)

        # y:
        # [[-2.251349   -3.8766491  -0.13464898]
        # [-0.48770458 -3.3176045  -1.0506046 ]]


    """

    def __init__(
        self, dim: Optional[int] = 1,
    ):
        super().__init__()

        self.dim = dim
        self._softmax_op = flow.builtin_op("softmax").Input("in").Output("out").Build()
        self._log_op = flow.builtin_op("log").Input("x").Output("y").Build()
        self._transpose_op = flow.builtin_op("transpose").Input("input").Output("output")
    

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, x):
        need_transpose, permute = _softmax_need_transpose(x, self.dim)

        if need_transpose:
            self._transpose_op = self._transpose_op.Attr("perm", permute).Build()
            x = self._transpose_op(x)[0]
        res = self._softmax_op(x)[0]
        res = self._log_op(res)[0]
        if need_transpose:
            res = transpose(res, perm=permute)
        
        return res

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

        

if __name__ == "__main__":
    flow.enable_eager_execution(True)
    import numpy as np

    x = flow.Tensor(np.array([[1, 2, 1, 5, 4]]))
    out = Softmax(1)(x)
    print(out.numpy())
