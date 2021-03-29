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
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

__all__ = ["SqueezeNet1_0", "SqueezeNet1_1"]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check

class MakeFireConv(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 padding=0,
                 name=None):
        super(MakeFireConv, self).__init__()
        self._conv = Conv2D(
            input_channels,
            output_channels,
            filter_size,
            padding=padding,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=ParamAttr(name=name + "_offset"))

    def forward(self, x):
        x = self._conv(x)
        x = F.relu(x)
        return x


class MakeFire(nn.Layer):
    def __init__(self,
                 input_channels,
                 squeeze_channels,
                 expand1x1_channels,
                 expand3x3_channels,
                 name=None):
        super(MakeFire, self).__init__()
        self._conv = MakeFireConv(
            input_channels, squeeze_channels, 1, name=name + "_squeeze1x1")
        self._conv_path1 = MakeFireConv(
            squeeze_channels, expand1x1_channels, 1, name=name + "_expand1x1")
        self._conv_path2 = MakeFireConv(
            squeeze_channels,
            expand3x3_channels,
            3,
            padding=1,
            name=name + "_expand3x3")

    def forward(self, inputs):
        x = self._conv(inputs)
        x1 = self._conv_path1(x)
        x2 = self._conv_path2(x)
        return paddle.concat([x1, x2], axis=1)


class SqueezeNet(nn.Layer):
    def __init__(self, version, class_dim=1000):
        super(SqueezeNet, self).__init__()
        self.version = version

        if self.version == "1.0":
            self._conv = Conv2D(
                3,
                96,
                7,
                stride=2,
                weight_attr=ParamAttr(name="conv1_weights"),
                bias_attr=ParamAttr(name="conv1_offset"))
            self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
            self._conv1 = MakeFire(96, 16, 64, 64, name="fire2")
            self._conv2 = MakeFire(128, 16, 64, 64, name="fire3")
            self._conv3 = MakeFire(128, 32, 128, 128, name="fire4")

            self._conv4 = MakeFire(256, 32, 128, 128, name="fire5")
            self._conv5 = MakeFire(256, 48, 192, 192, name="fire6")
            self._conv6 = MakeFire(384, 48, 192, 192, name="fire7")
            self._conv7 = MakeFire(384, 64, 256, 256, name="fire8")

            self._conv8 = MakeFire(512, 64, 256, 256, name="fire9")
        else:
            self._conv = Conv2D(
                3,
                64,
                3,
                stride=2,
                padding=1,
                weight_attr=ParamAttr(name="conv1_weights"),
                bias_attr=ParamAttr(name="conv1_offset"))
            self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
            self._conv1 = MakeFire(64, 16, 64, 64, name="fire2")
            self._conv2 = MakeFire(128, 16, 64, 64, name="fire3")

            self._conv3 = MakeFire(128, 32, 128, 128, name="fire4")
            self._conv4 = MakeFire(256, 32, 128, 128, name="fire5")

            self._conv5 = MakeFire(256, 48, 192, 192, name="fire6")
            self._conv6 = MakeFire(384, 48, 192, 192, name="fire7")
            self._conv7 = MakeFire(384, 64, 256, 256, name="fire8")
            self._conv8 = MakeFire(512, 64, 256, 256, name="fire9")

        self._drop = Dropout(p=0.5, mode="downscale_in_infer")
        self._conv9 = Conv2D(
            512,
            class_dim,
            1,
            weight_attr=ParamAttr(name="conv10_weights"),
            bias_attr=ParamAttr(name="conv10_offset"))
        self._avg_pool = AdaptiveAvgPool2D(1)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = F.relu(x)
        x = self._pool(x)
        if self.version == "1.0":
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._conv3(x)
            x = self._pool(x)
            x = self._conv4(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._pool(x)
            x = self._conv8(x)
        else:
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._pool(x)
            x = self._conv3(x)
            x = self._conv4(x)
            x = self._pool(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._conv8(x)
        x = self._drop(x)
        x = self._conv9(x)
        x = F.relu(x)
        x = self._avg_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        return x


def SqueezeNet1_0(**args):
    model = SqueezeNet(version="1.0", **args)
    return model


def SqueezeNet1_1(**args):
    model = SqueezeNet(version="1.1", **args)
    return model

def test_SqueezeNet1_0(test_case):
    load_paddle_module_and_check(
        test_case, SqueezeNet1_0, input_size=(1, 3, 224, 224), train_flag=False,
    )
