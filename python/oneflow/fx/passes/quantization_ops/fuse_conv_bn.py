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

        self.conv_weight = conv_module.weight
        self.conv_bias = conv_module.bias
        self.conv_out_channels = conv_module.out_channels
        self.conv_stride = conv_module.stride
        self.conv_padding = conv_module.padding
        self.conv_dilation = conv_module.dilation
        self.conv_groups = conv_module.groups

        self.bn_running_mean = bn_module.running_mean
        self.bn_running_var = bn_module.running_var
        self.bn_weight = bn_module.weight
        self.bn_bias = bn_module.bias
        self.bn_affine = bn_module.affine
        self.bn_eps = bn_module.eps

        # self.register_buffer("new_zero_1", flow.Tensor(1))
        # self.new_zero_1.fill_(0)
        # self.register_buffer("new_zero_2", flow.Tensor(1))
        # self.new_zero_2.fill_(0)

        self.moving_min_max_observer_1 = flow.nn.MovingAverageMinMaxObserver(
            training=self.training,
            quantization_formula=quantization_formula,
            stop_update_after_iters=1,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            momentum=momentum,
        )

        self.moving_min_max_observer_2 = flow.nn.MovingAverageMinMaxObserver(
            training=self.training,
            quantization_formula=quantization_formula,
            stop_update_after_iters=1,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            momentum=momentum,
        )

        self.fake_quantization = flow.nn.FakeQuantization(
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
        )

    def fold_bn(self, mean, std):
        if self.bn_affine:
            gamma_ = self.bn_weight / std
            weight = self.conv_weight * gamma_.view(self.conv_out_channels, 1, 1, 1)
            if self.conv_bias is not None:
                bias = gamma_ * self.conv_bias - gamma_ * mean + self.bn_bias
            else:
                bias = self.bn_bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_weight * gamma_
            if self.conv_bias is not None:
                bias = gamma_ * self.conv_bias - gamma_ * mean
            else:
                bias = -gamma_ * mean

        return weight, bias

    def forward(self, x):
        new_zero_1 = flow.zeros(1)
        new_zero_1 = new_zero_1.to(x.device.type)
        scale, zero_point = self.moving_min_max_observer_1(x, new_zero_1)
        x = self.fake_quantization(x, scale, zero_point)
        if self.training:
            y = flow.nn.functional.conv2d(
                x,
                self.conv_weight,
                self.conv_bias,
                stride=self.conv_stride,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                groups=self.conv_groups,
            )
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.view(self.conv_out_channels, -1)  # CNHW -> C,NHW
            mean = y.mean(1)
            var = y.var(1)
            with flow.no_grad():
                self.bn_running_mean = (
                    self.bn_momentum * self.bn_running_mean
                    + (1 - self.bn_momentum) * mean
                )
                self.bn_running_var = (
                    self.bn_momentum * self.bn_running_var
                    + (1 - self.bn_momentum) * var
                )
        else:
            mean = self.bn_running_mean + 0.0
            mean = mean.to(x.device)
            var = self.bn_running_var + 0.0
            var = var.to(x.device)

        std = flow.sqrt(var + self.bn_eps)
        weight, bias = self.fold_bn(mean, std)

        # weight_scale, weight_zero_point = flow.ones(1), flow.zeros(1)
        # weight_scale = weight_scale.to(x.device.type)
        # weight_zero_point = weight_zero_point.to(x.device.type)
        new_zero_2 = flow.zeros(1)
        new_zero_2 = new_zero_2.to(x.device.type)
        weight_scale, weight_zero_point = self.moving_min_max_observer_2(
            weight, new_zero_2
        )
        res = flow.nn.functional.conv2d(
            x,
            self.fake_quantization(weight, weight_scale, weight_zero_point),
            bias,
            stride=self.conv_stride,
            padding=self.conv_padding,
            dilation=self.conv_dilation,
            groups=self.conv_groups,
        )
        return res
