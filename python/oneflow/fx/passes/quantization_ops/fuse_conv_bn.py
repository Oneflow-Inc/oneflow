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

__all__ = ["QConvBN"]


class QConvBN(flow.nn.Module):
    def __init__(
        self,
        conv_module,
        bn_module,
        quantization_bit=8,
        quantization_scheme="symmetric",
        quantization_formula="google",
        per_layer_quantization=True,
        momentum=0.95,
    ):
        super().__init__()
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.conv_module = conv_module
        self.bn_module = bn_module

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

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(
                self.conv_module.out_channels, 1, 1, 1
            )
            if self.conv_module.bias is not None:
                bias = (
                    gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
                )
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean

        return weight, bias

    def forward(self, x):
        scale, zero_point = self.moving_min_max_observer(
            x, flow.tensor([0], dtype=flow.int64).to(x.device.type)
        )
        x = self.fake_quantization(x, scale, zero_point)
        if self.training:
            y = flow.nn.functional.conv2d(
                x,
                self.conv_module.weight,
                self.conv_module.bias,
                stride=self.conv_module.stride,
                padding=self.conv_module.padding,
                dilation=self.conv_module.dilation,
                groups=self.conv_module.groups,
            )
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.view(self.conv_module.out_channels, -1)  # CNHW -> C,NHW
            mean = y.mean(1)
            var = y.var(1)
            with flow.no_grad():
                self.bn_module.running_mean = (
                    self.bn_module.momentum * self.bn_module.running_mean
                    + (1 - self.bn_module.momentum) * mean
                )
                self.bn_module.running_var = (
                    self.bn_module.momentum * self.bn_module.running_var
                    + (1 - self.bn_module.momentum) * var
                )
        else:
            mean = flow.tensor(self.bn_module.running_mean)
            var = flow.tensor(self.bn_module.running_var)

        std = flow.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_bn(mean, std)

        weight_scale, weight_zero_point = self.min_max_observer(weight)
        res = flow.nn.functional.conv2d(
            x,
            self.fake_quantization(weight, weight_scale, weight_zero_point),
            bias,
            stride=self.conv_module.stride,
            padding=self.conv_module.padding,
            dilation=self.conv_module.dilation,
            groups=self.conv_module.groups,
        )
        return res
