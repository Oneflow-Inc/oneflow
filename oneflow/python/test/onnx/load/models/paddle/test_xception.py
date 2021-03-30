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
import sys

__all__ = ["Xception41", "Xception65", "Xception71"]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
    ):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
        )
        bn_name = "bn_" + name
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
        )

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SeparableConv(nn.Layer):
    def __init__(self, input_channels, output_channels, stride=1, name=None):
        super(SeparableConv, self).__init__()

        self._pointwise_conv = ConvBNLayer(
            input_channels, output_channels, 1, name=name + "_sep"
        )
        self._depthwise_conv = ConvBNLayer(
            output_channels,
            output_channels,
            3,
            stride=stride,
            groups=output_channels,
            name=name + "_dw",
        )

    def forward(self, inputs):
        x = self._pointwise_conv(inputs)
        x = self._depthwise_conv(x)
        return x


class EntryFlowBottleneckBlock(nn.Layer):
    def __init__(
        self, input_channels, output_channels, stride=2, name=None, relu_first=False
    ):
        super(EntryFlowBottleneckBlock, self).__init__()
        self.relu_first = relu_first

        self._short = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            weight_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False,
        )
        self._conv1 = SeparableConv(
            input_channels, output_channels, stride=1, name=name + "_branch2a_weights"
        )
        self._conv2 = SeparableConv(
            output_channels, output_channels, stride=1, name=name + "_branch2b_weights"
        )
        self._pool = MaxPool2D(kernel_size=3, stride=stride, padding=1)

    def forward(self, inputs):
        conv0 = inputs
        short = self._short(inputs)
        if self.relu_first:
            conv0 = F.relu(conv0)
        conv1 = self._conv1(conv0)
        conv2 = F.relu(conv1)
        conv2 = self._conv2(conv2)
        pool = self._pool(conv2)
        return paddle.add(x=short, y=pool)


class EntryFlow(nn.Layer):
    def __init__(self, block_num=3):
        super(EntryFlow, self).__init__()

        name = "entry_flow"
        self.block_num = block_num
        self._conv1 = ConvBNLayer(3, 32, 3, stride=2, act="relu", name=name + "_conv1")
        self._conv2 = ConvBNLayer(32, 64, 3, act="relu", name=name + "_conv2")
        if block_num == 3:
            self._conv_0 = EntryFlowBottleneckBlock(
                64, 128, stride=2, name=name + "_0", relu_first=False
            )
            self._conv_1 = EntryFlowBottleneckBlock(
                128, 256, stride=2, name=name + "_1", relu_first=True
            )
            self._conv_2 = EntryFlowBottleneckBlock(
                256, 728, stride=2, name=name + "_2", relu_first=True
            )
        elif block_num == 5:
            self._conv_0 = EntryFlowBottleneckBlock(
                64, 128, stride=2, name=name + "_0", relu_first=False
            )
            self._conv_1 = EntryFlowBottleneckBlock(
                128, 256, stride=1, name=name + "_1", relu_first=True
            )
            self._conv_2 = EntryFlowBottleneckBlock(
                256, 256, stride=2, name=name + "_2", relu_first=True
            )
            self._conv_3 = EntryFlowBottleneckBlock(
                256, 728, stride=1, name=name + "_3", relu_first=True
            )
            self._conv_4 = EntryFlowBottleneckBlock(
                728, 728, stride=2, name=name + "_4", relu_first=True
            )
        else:
            sys.exit(-1)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)

        if self.block_num == 3:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
        elif self.block_num == 5:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
            x = self._conv_3(x)
            x = self._conv_4(x)
        return x


class MiddleFlowBottleneckBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, name):
        super(MiddleFlowBottleneckBlock, self).__init__()

        self._conv_0 = SeparableConv(
            input_channels, output_channels, stride=1, name=name + "_branch2a_weights"
        )
        self._conv_1 = SeparableConv(
            output_channels, output_channels, stride=1, name=name + "_branch2b_weights"
        )
        self._conv_2 = SeparableConv(
            output_channels, output_channels, stride=1, name=name + "_branch2c_weights"
        )

    def forward(self, inputs):
        conv0 = F.relu(inputs)
        conv0 = self._conv_0(conv0)
        conv1 = F.relu(conv0)
        conv1 = self._conv_1(conv1)
        conv2 = F.relu(conv1)
        conv2 = self._conv_2(conv2)
        return paddle.add(x=inputs, y=conv2)


class MiddleFlow(nn.Layer):
    def __init__(self, block_num=8):
        super(MiddleFlow, self).__init__()

        self.block_num = block_num
        self._conv_0 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_0")
        self._conv_1 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_1")
        self._conv_2 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_2")
        self._conv_3 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_3")
        self._conv_4 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_4")
        self._conv_5 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_5")
        self._conv_6 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_6")
        self._conv_7 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_7")
        if block_num == 16:
            self._conv_8 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_8")
            self._conv_9 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_9")
            self._conv_10 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_10")
            self._conv_11 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_11")
            self._conv_12 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_12")
            self._conv_13 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_13")
            self._conv_14 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_14")
            self._conv_15 = MiddleFlowBottleneckBlock(728, 728, name="middle_flow_15")

    def forward(self, inputs):
        x = self._conv_0(inputs)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = self._conv_4(x)
        x = self._conv_5(x)
        x = self._conv_6(x)
        x = self._conv_7(x)
        if self.block_num == 16:
            x = self._conv_8(x)
            x = self._conv_9(x)
            x = self._conv_10(x)
            x = self._conv_11(x)
            x = self._conv_12(x)
            x = self._conv_13(x)
            x = self._conv_14(x)
            x = self._conv_15(x)
        return x


class ExitFlowBottleneckBlock(nn.Layer):
    def __init__(self, input_channels, output_channels1, output_channels2, name):
        super(ExitFlowBottleneckBlock, self).__init__()

        self._short = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels2,
            kernel_size=1,
            stride=2,
            padding=0,
            weight_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False,
        )
        self._conv_1 = SeparableConv(
            input_channels, output_channels1, stride=1, name=name + "_branch2a_weights"
        )
        self._conv_2 = SeparableConv(
            output_channels1,
            output_channels2,
            stride=1,
            name=name + "_branch2b_weights",
        )
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        short = self._short(inputs)
        conv0 = F.relu(inputs)
        conv1 = self._conv_1(conv0)
        conv2 = F.relu(conv1)
        conv2 = self._conv_2(conv2)
        pool = self._pool(conv2)
        return paddle.add(x=short, y=pool)


class ExitFlow(nn.Layer):
    def __init__(self, class_dim):
        super(ExitFlow, self).__init__()

        name = "exit_flow"

        self._conv_0 = ExitFlowBottleneckBlock(728, 728, 1024, name=name + "_1")
        self._conv_1 = SeparableConv(1024, 1536, stride=1, name=name + "_2")
        self._conv_2 = SeparableConv(1536, 2048, stride=1, name=name + "_3")
        self._pool = AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self._out = Linear(
            2048,
            class_dim,
            weight_attr=ParamAttr(name="fc_weights", initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"),
        )

    def forward(self, inputs):
        conv0 = self._conv_0(inputs)
        conv1 = self._conv_1(conv0)
        conv1 = F.relu(conv1)
        conv2 = self._conv_2(conv1)
        conv2 = F.relu(conv2)
        pool = self._pool(conv2)
        pool = paddle.flatten(pool, start_axis=1, stop_axis=-1)
        out = self._out(pool)
        return out


class Xception(nn.Layer):
    def __init__(self, entry_flow_block_num=3, middle_flow_block_num=8, class_dim=1000):
        super(Xception, self).__init__()
        self.entry_flow_block_num = entry_flow_block_num
        self.middle_flow_block_num = middle_flow_block_num
        self._entry_flow = EntryFlow(entry_flow_block_num)
        self._middle_flow = MiddleFlow(middle_flow_block_num)
        self._exit_flow = ExitFlow(class_dim)

    def forward(self, inputs):
        x = self._entry_flow(inputs)
        x = self._middle_flow(x)
        x = self._exit_flow(x)
        return x


def Xception41(**args):
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=8, **args)
    return model


def Xception65(**args):
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=16, **args)
    return model


def Xception71(**args):
    model = Xception(entry_flow_block_num=5, middle_flow_block_num=16, **args)
    return model


def test_Xception41(test_case):
    load_paddle_module_and_check(
        test_case, Xception41, input_size=(1, 3, 224, 224), train_flag=False,
    )


from absl import app
from absl.testing import absltest

test_case = absltest.TestCase
test_Xception41(test_case)
