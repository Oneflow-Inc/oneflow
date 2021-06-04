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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.nn.module import Module
from oneflow.python.nn.init import _calculate_fan_in_and_fan_out
from typing import Optional, List, Tuple
import math


@oneflow_export("nn.Identity")
@experimental_api
class Identity(Module):
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow

        m = flow.nn.Identity()
        input = flow.Tensor(np.random.rand(2, 3, 4, 5))

        output = m(input)

        # output = input

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


@oneflow_export("nn.Linear")
@experimental_api
class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        
        - in_features: size of each input sample

        - out_features: size of each output sample

        - bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = {in\_features}`
        
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = {out\_features}`.

    Attr:
        - :attr:`weight`: the learnable weights of the module of shape :math:`({out\_features}, {in\_features})`. The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`(k = 1 / {in\_features})`
        
        - :attr:`bias`: the learnable bias of the module of shape :math:`({out\_features})`. If :attr:`bias` is ``True``, the values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`(k = 1 / {in\_features})`

    
    For example: 

    .. code-block:: python 

        >>> import numpy as np
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()


        >>> m = flow.nn.Linear(20, 30, False)
        >>> input = flow.Tensor(np.random.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
        flow.Size([128, 30])

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.use_bias = bias
        self.weight = flow.nn.Parameter(flow.Tensor(out_features, in_features))
        self.bias = None

        if bias:
            self.bias = flow.nn.Parameter(flow.Tensor(out_features))

        self._matmul_op = (
            flow.builtin_op("matmul")
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", False)
            .Attr("transpose_b", True)
            .Attr("alpha", 1.0)
            .Build()
        )

        self._broadcast_matmul_op = (
            flow.builtin_op("broadcast_matmul")
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", False)
            .Attr("transpose_b", True)
            .Attr("alpha", 1.0)
            .Build()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        flow.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            flow.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert len(x.shape) >= 2, "Tensor x's dim should >=2"

        if len(x.shape) == 2:
            res = self._matmul_op(x, self.weight)[0]
        else:
            res = self._broadcast_matmul_op(x, self.weight)[0]

        if self.use_bias:
            res += self.bias

        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
