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

__all__ = ["QLinear"]


class QLinear(flow.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        quantization_bit=8,
        quantization_scheme="symmetric",
        quantization_formula="google",
        per_layer_quantization=True,
        momentum = 0.95,
    ) -> None:
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.moving_min_max_observer = flow.nn.MovingAverageMinMaxObserver(training=self.training, quantization_formula=quantization_formula,
                                                                       stop_update_after_iters=1, quantization_bit=quantization_bit,
                                                                       quantization_scheme=quantization_scheme, momentum=momentum)
        self.min_max_observer = flow.nn.MinMaxObserver(
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            per_layer_quantization=per_layer_quantization,
        )
        self.fake_quantization = flow.nn.FakeQuantization(
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
        )

    def forward(self, x):
        scale, zero_point = self.moving_min_max_observer(x, flow.tensor([0], dtype=flow.int64).to(x.device.type))
        x = self.fake_quantization(x, scale, zero_point)
        weight_scale, weight_zero_point = self.min_max_observer(self.weight)
        self.weight = flow.nn.Parameter(
            self.fake_quantization(self.weight, weight_scale, weight_zero_point)
        )
        res = flow._C.matmul(x, self.weight, transpose_a=False, transpose_b=True)
        if self.bias is not None:
            res += self.bias
        return res
