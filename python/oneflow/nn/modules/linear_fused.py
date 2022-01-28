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
import math

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.init import _calculate_fan_in_and_fan_out
from oneflow.nn.module import Module

class FusedLinearReLU(Module):
    """Applies a linear transformation with relu activation to the incoming data: :math:`y = ReLU(xA^T + b)`

    Currently only support in GPU. 

    Args:
        in_features: size of each input sample

        out_features: size of each output sample

        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = {in\\_features}`

        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = {out\\_features}`.

    Attr:
        - :attr:`weight`: the learnable weights of the module of shape :math:`({out\\_features}, {in\\_features})`. The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where :math:`(k = 1 / {in\\_features})`

        - :attr:`bias`: the learnable bias of the module of shape :math:`({out\\_features})`. If :attr:`bias` is ``True``, the values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where :math:`(k = 1 / {in\\_features})`


    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        

        >>> m = flow.nn.FusedLinearReLU(20, 30).to("cuda")
        >>> input = flow.Tensor(np.random.randn(128, 20)).to("cuda")
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([128, 30])

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = flow.nn.Parameter(flow.Tensor(out_features, in_features))
        self.bias = flow.nn.Parameter(flow.Tensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        flow.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            flow.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        res = flow._C.fused_matmul_bias_add_relu(x, self.weight, self.bias, transpose_a=False, transpose_b=True)
        return res

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
