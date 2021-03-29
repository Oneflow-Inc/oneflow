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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

__all__ = [
    "Res2Net50_48w_2s", "Res2Net50_26w_4s", "Res2Net50_14w_8s",
    "Res2Net50_48w_2s", "Res2Net50_26w_6s", "Res2Net50_26w_8s",
    "Res2Net101_26w_4s", "Res2Net152_26w_4s", "Res2Net200_26w_4s"
]

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
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels1,
                 num_channels2,
                 num_filters,
                 stride,
                 scales,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        self.scales = scales
        self.conv0 = ConvBNLayer(
            num_channels=num_channels1,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1_list = []
        for s in range(scales - 1):
            conv1 = self.add_sublayer(
                name + '_branch2b_' + str(s + 1),
                ConvBNLayer(
                    num_channels=num_filters // scales,
                    num_filters=num_filters // scales,
                    filter_size=3,
                    stride=stride,
                    act='relu',
                    name=name + '_branch2b_' + str(s + 1)))
            self.conv1_list.append(conv1)
        self.pool2d_avg = AvgPool2D(kernel_size=3, stride=stride, padding=1)

        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_channels2,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels1,
                num_filters=num_channels2,
                filter_size=1,
                stride=stride,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        xs = paddle.split(y, self.scales, 1)
        ys = []
        for s, conv1 in enumerate(self.conv1_list):
            if s == 0 or self.stride == 2:
                ys.append(conv1(xs[s]))
            else:
                ys.append(conv1(paddle.add(xs[s], ys[-1])))
        if self.stride == 1:
            ys.append(xs[-1])
        else:
            ys.append(self.pool2d_avg(xs[-1]))
        conv1 = paddle.concat(ys, axis=1)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class Res2Net(nn.Layer):
    def __init__(self, layers=50, scales=4, width=26, class_dim=1000):
        super(Res2Net, self).__init__()

        self.layers = layers
        self.scales = scales
        self.width = width
        basic_width = self.width * self.scales
        supported_layers = [50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024]
        num_channels2 = [256, 512, 1024, 2048]
        num_filters = [basic_width * t for t in [1, 2, 4, 8]]

        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1")
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels1=num_channels[block]
                        if i == 0 else num_channels2[block],
                        num_channels2=num_channels2[block],
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        scales=scales,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name))
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def Res2Net50_48w_2s(**args):
    model = Res2Net(layers=50, scales=2, width=48, **args)
    return model


def Res2Net50_26w_4s(**args):
    model = Res2Net(layers=50, scales=4, width=26, **args)
    return model


def Res2Net50_14w_8s(**args):
    model = Res2Net(layers=50, scales=8, width=14, **args)
    return model


def Res2Net50_26w_6s(**args):
    model = Res2Net(layers=50, scales=6, width=26, **args)
    return model


def Res2Net50_26w_8s(**args):
    model = Res2Net(layers=50, scales=8, width=26, **args)
    return model


def Res2Net101_26w_4s(**args):
    model = Res2Net(layers=101, scales=4, width=26, **args)
    return model


def Res2Net152_26w_4s(**args):
    model = Res2Net(layers=152, scales=4, width=26, **args)
    return model


def Res2Net200_26w_4s(**args):
    model = Res2Net(layers=200, scales=4, width=26, **args)
    return model

def test_Res2Net200_26w_4s(test_case):
    load_paddle_module_and_check(
        test_case, Res2Net50_48w_2s, input_size=(1, 3, 224, 224), train_flag=False,
    )

def test_Res2Net200_26w_4s(test_case):
    load_paddle_module_and_check(
        test_case, Res2Net50_48w_2s, input_size=(1, 3, 224, 224), train_flag=False,
    )

from absl import app
from absl.testing import absltest

test_case = absltest.TestCase
test_Res2Net200_26w_4s(test_case)
