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
from oneflow.nn.modules.module import Module
import os


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


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:

        - in_features: size of each input sample

        - out_features: size of each output sample

        - bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``

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


        >>> m = flow.nn.Linear(20, 30, False)
        >>> input = flow.Tensor(np.random.randn(128, 20))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([128, 30])

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = flow.nn.Parameter(
            flow.Tensor(out_features, in_features).to(dtype=dtype, device=device)
        )
        self.bias = (
            flow.nn.Parameter(flow.Tensor(out_features).to(dtype=dtype, device=device))
            if bias
            else None
        )
        self.use_fused_matmul_bias = (
            self.bias is not None
            and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1"
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") == "1":
            return
        flow.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            flow.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.use_fused_matmul_bias:
            return flow._C.fused_matmul_bias(x, self.weight, self.bias)
        else:
            res = flow._C.matmul(x, self.weight, transpose_a=False, transpose_b=True)
            if self.bias is not None:
                res += self.bias
            return res

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.random.randn(128, 20))
        >>> weight = flow.tensor(np.random.randn(30, 20))
        >>> output = flow.nn.functional.linear(input, weight)
        >>> output.size()
        oneflow.Size([128, 30])

    """
    res = flow._C.matmul(input, weight, transpose_a=False, transpose_b=True)
    if bias is not None:
        res += bias
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
