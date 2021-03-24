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
from paddle.nn.initializer import Uniform
import math

__all__ = ["DarkNet53"]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class ConvBNLayer(nn.Layer):
    def __init__(
        self, input_channels, output_channels, filter_size, stride, padding, name=None
    ):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False,
        )

        bn_name = name + ".bn"
        self._bn = BatchNorm(
            num_channels=output_channels,
            act="relu",
            param_attr=ParamAttr(name=bn_name + ".scale"),
            bias_attr=ParamAttr(name=bn_name + ".offset"),
            moving_mean_name=bn_name + ".mean",
            moving_variance_name=bn_name + ".var",
        )

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class BasicBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, name=None):
        super(BasicBlock, self).__init__()

        self._conv1 = ConvBNLayer(
            input_channels, output_channels, 1, 1, 0, name=name + ".0"
        )
        self._conv2 = ConvBNLayer(
            output_channels, output_channels * 2, 3, 1, 1, name=name + ".1"
        )

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        return paddle.add(x=inputs, y=x)


class DarkNet(nn.Layer):
    def __init__(self, class_dim=1000):
        super(DarkNet, self).__init__()

        self.stages = [1, 2, 8, 8, 4]
        self._conv1 = ConvBNLayer(3, 32, 3, 1, 1, name="yolo_input")
        self._conv2 = ConvBNLayer(32, 64, 3, 2, 1, name="yolo_input.downsample")

        self._basic_block_01 = BasicBlock(64, 32, name="stage.0.0")
        self._downsample_0 = ConvBNLayer(64, 128, 3, 2, 1, name="stage.0.downsample")

        self._basic_block_11 = BasicBlock(128, 64, name="stage.1.0")
        self._basic_block_12 = BasicBlock(128, 64, name="stage.1.1")
        self._downsample_1 = ConvBNLayer(128, 256, 3, 2, 1, name="stage.1.downsample")

        self._basic_block_21 = BasicBlock(256, 128, name="stage.2.0")
        self._basic_block_22 = BasicBlock(256, 128, name="stage.2.1")
        self._basic_block_23 = BasicBlock(256, 128, name="stage.2.2")
        self._basic_block_24 = BasicBlock(256, 128, name="stage.2.3")
        self._basic_block_25 = BasicBlock(256, 128, name="stage.2.4")
        self._basic_block_26 = BasicBlock(256, 128, name="stage.2.5")
        self._basic_block_27 = BasicBlock(256, 128, name="stage.2.6")
        self._basic_block_28 = BasicBlock(256, 128, name="stage.2.7")
        self._downsample_2 = ConvBNLayer(256, 512, 3, 2, 1, name="stage.2.downsample")

        self._basic_block_31 = BasicBlock(512, 256, name="stage.3.0")
        self._basic_block_32 = BasicBlock(512, 256, name="stage.3.1")
        self._basic_block_33 = BasicBlock(512, 256, name="stage.3.2")
        self._basic_block_34 = BasicBlock(512, 256, name="stage.3.3")
        self._basic_block_35 = BasicBlock(512, 256, name="stage.3.4")
        self._basic_block_36 = BasicBlock(512, 256, name="stage.3.5")
        self._basic_block_37 = BasicBlock(512, 256, name="stage.3.6")
        self._basic_block_38 = BasicBlock(512, 256, name="stage.3.7")
        self._downsample_3 = ConvBNLayer(512, 1024, 3, 2, 1, name="stage.3.downsample")

        self._basic_block_41 = BasicBlock(1024, 512, name="stage.4.0")
        self._basic_block_42 = BasicBlock(1024, 512, name="stage.4.1")
        self._basic_block_43 = BasicBlock(1024, 512, name="stage.4.2")
        self._basic_block_44 = BasicBlock(1024, 512, name="stage.4.3")

        self._pool = AdaptiveAvgPool2D(1)

        stdv = 1.0 / math.sqrt(1024.0)
        self._out = Linear(
            1024,
            class_dim,
            weight_attr=ParamAttr(name="fc_weights", initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"),
        )

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)

        x = self._basic_block_01(x)
        x = self._downsample_0(x)

        x = self._basic_block_11(x)
        x = self._basic_block_12(x)
        x = self._downsample_1(x)

        x = self._basic_block_21(x)
        x = self._basic_block_22(x)
        x = self._basic_block_23(x)
        x = self._basic_block_24(x)
        x = self._basic_block_25(x)
        x = self._basic_block_26(x)
        x = self._basic_block_27(x)
        x = self._basic_block_28(x)
        x = self._downsample_2(x)

        x = self._basic_block_31(x)
        x = self._basic_block_32(x)
        x = self._basic_block_33(x)
        x = self._basic_block_34(x)
        x = self._basic_block_35(x)
        x = self._basic_block_36(x)
        x = self._basic_block_37(x)
        x = self._basic_block_38(x)
        x = self._downsample_3(x)

        x = self._basic_block_41(x)
        x = self._basic_block_42(x)
        x = self._basic_block_43(x)
        x = self._basic_block_44(x)

        x = self._pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self._out(x)
        return x


def DarkNet53(**args):
    model = DarkNet(**args)
    return model


def test_darknet(test_case):
    load_paddle_module_and_check(
        test_case, DarkNet53, input_size=(1, 3, 224, 224), train_flag=False,
    )
