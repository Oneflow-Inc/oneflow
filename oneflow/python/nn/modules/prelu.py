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


@oneflow_export("nn.PReLU")
@experimental_api
class PReLU(Module):
    """Applies the element-wise function:

    .. math::
        PReLU(x) = \max(0,x) + a * \min(0,x)

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Attr:
        - weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. code-block:: python

        import oneflow.experimental as flow

        m = nn.PReLU()
        input = flow.randn(2)
        output = m(input)

    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = flow.nn.Parameter(flow.Tensor(num_parameters, 1, 1).fill_(init))
        self.op = flow.builtin_op("prelu").Input("x").Input("alpha").Output("y").Build()

    def forward(self, x):
        assert (
            self.num_parameters == 1 or self.num_parameters == x.shape[1]
        ), f"num_parameters in prelu must be 1 or {x.shape[1]}"
        return self.op(x, self.weight)[0]