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

__all__ = ["DConv2d"]


class DConv2d(flow.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        quantization_bit=8,
        quantization_scheme="symmetric",
        quantization_formula="google",
        per_layer_quantization=True,
        momentum=0.95,
    ) -> None:
        super(DConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups
        )

        self.moving_min_max_observer = flow.nn.MovingAverageMinMaxObserver(
            training=self.training,
            quantization_formula=quantization_formula,
            stop_update_after_iters=1,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            momentum=momentum,
        )

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

        weight_scale, weight_zero_point = self.min_max_observer(self.weight)
        self.weight = flow.nn.Parameter(
            self.fake_quantization(self.weight, weight_scale, weight_zero_point)
        )
        self.register_buffer("new_zero", flow.Tensor(1))
        self.new_zero.fill_(0)

    def forward(self, x):
        scale, zero_point = self.moving_min_max_observer(
            x, self.new_zero.to(flow.int64).to(x.device.type)
        )
        x = self.fake_quantization(x, scale, zero_point)
        return flow.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
