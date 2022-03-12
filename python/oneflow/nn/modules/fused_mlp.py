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
from typing import Tuple


class FusedMLP(Module):
    """Applies a linear transformation with relu activation to the incoming data: :math:`y = ReLU(xA^T + b)`

    Args:
        in_features: size of each input sample

        hidden_features: A tuple of each Linear layer hidden size

        out_features: The final Linear layer hidden size

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = {in\\_features}`

        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = {out\\_features}`.

    Attr:
        - :attr:`skip_final_activation`: Whether to skip final hidden layer's activation. Default: False. 

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        

        >>> m = flow.nn.FusedMLP(128, [256, 512], 1024).to("cuda")
        >>> input = flow.Tensor(np.random.randn(1, 128)).to("cuda")
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 1024])

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Tuple[int],
        out_features: int,
        skip_final_activation=False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        # TODO(zzk): Add more activation support.
        self.skip_final_activation = skip_final_activation
        self.hidden_layer_num = len(hidden_features)

        self.add_parameters()
        self.reset_parameters()

    def add_parameters(self) -> None:
        if self.hidden_layer_num != 0:
            # First layer.
            self.register_parameter(
                f"weight_{0}",
                flow.nn.Parameter(
                    flow.Tensor(self.hidden_features[0], self.in_features)
                ),
            )
            self.register_parameter(
                f"bias_{0}", flow.nn.Parameter(flow.Tensor(self.hidden_features[0]))
            )

            # Middle Layer.
            for idx in range(1, self.hidden_layer_num):
                self.register_parameter(
                    f"weight_{idx}",
                    flow.nn.Parameter(
                        flow.Tensor(
                            self.hidden_features[idx], self.hidden_features[idx - 1],
                        )
                    ),
                )
                self.register_parameter(
                    f"bias_{idx}",
                    flow.nn.Parameter(flow.Tensor(self.hidden_features[idx])),
                )

            # Final Layer.
            self.register_parameter(
                f"weight_{self.hidden_layer_num}",
                flow.nn.Parameter(
                    flow.Tensor(
                        self.out_features,
                        self.hidden_features[self.hidden_layer_num - 1],
                    )
                ),
            )
            self.register_parameter(
                f"bias_{self.hidden_layer_num}",
                flow.nn.Parameter(flow.Tensor(self.out_features)),
            )
        else:
            # there is only 1 layer.
            self.register_parameter(
                f"weight_{0}",
                flow.nn.Parameter(flow.Tensor(self.out_features, self.in_features)),
            )
            self.register_parameter(
                f"bias_{0}", flow.nn.Parameter(flow.Tensor(self.out_features))
            )

    def weight(self, i):
        return getattr(self, f"weight_{i}")

    def weights(self):
        return [self.weight(i) for i in range(self.hidden_layer_num + 1)]

    def bias(self, i):
        return getattr(self, f"bias_{i}")

    def biases(self):
        return [self.bias(i) for i in range(self.hidden_layer_num + 1)]

    def reset_parameters(self) -> None:
        for layer_idx in range(self.hidden_layer_num + 1):
            flow.nn.init.kaiming_uniform_(self.weight(layer_idx), a=math.sqrt(5))
            (fan_in, _) = _calculate_fan_in_and_fan_out(self.weight(layer_idx))
            bound = 1 / math.sqrt(fan_in)
            flow.nn.init.uniform_(self.bias(layer_idx), -bound, bound)

    def forward(self, x):
        res = flow._C.fused_mlp(
            x, self.weights(), self.biases(), self.skip_final_activation
        )
        return res

    def extra_repr(self) -> str:
        return "in_features={}, hidden_features={}, out_features={}, skip_final_activation={}".format(
            self.in_features,
            self.hidden_features,
            self.out_features,
            self.skip_final_activation,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
