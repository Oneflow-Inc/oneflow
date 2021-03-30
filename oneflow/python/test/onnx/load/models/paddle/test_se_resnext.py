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

__all__ = ["SE_ResNeXt50_32x4d", "SE_ResNeXt101_32x4d", "SE_ResNeXt152_64x4d"]

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
        bn_name = name + "_bn"
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
        )

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        cardinality,
        reduction_ratio,
        shortcut=True,
        if_first=False,
        name=None,
    ):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            name="conv" + name + "_x1",
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            groups=cardinality,
            stride=stride,
            act="relu",
            name="conv" + name + "_x2",
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1,
            act=None,
            name="conv" + name + "_x3",
        )
        self.scale = SELayer(
            num_channels=num_filters * 2 if cardinality == 32 else num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            reduction_ratio=reduction_ratio,
            name="fc" + name,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2 if cardinality == 32 else num_filters,
                filter_size=1,
                stride=stride,
                name="conv" + name + "_prj",
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=scale)
        y = F.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_sqz_weights"
            ),
            bias_attr=ParamAttr(name=name + "_sqz_offset"),
        )
        self.relu = nn.ReLU()
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_exc_weights"
            ),
            bias_attr=ParamAttr(name=name + "_exc_offset"),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = self.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = self.sigmoid(excitation)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = input * excitation
        return out


class ResNeXt(nn.Layer):
    def __init__(self, layers=50, class_dim=1000, cardinality=32):
        super(ResNeXt, self).__init__()

        self.layers = layers
        self.cardinality = cardinality
        self.reduction_ratio = 16
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )
        supported_cardinality = [32, 64]
        assert (
            cardinality in supported_cardinality
        ), "supported cardinality is {} but input cardinality is {}".format(
            supported_cardinality, cardinality
        )
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = (
            [128, 256, 512, 1024] if cardinality == 32 else [256, 512, 1024, 2048]
        )
        if layers < 152:
            self.conv = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act="relu",
                name="conv1",
            )
        else:
            self.conv1_1 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=3,
                stride=2,
                act="relu",
                name="conv1",
            )
            self.conv1_2 = ConvBNLayer(
                num_channels=64,
                num_filters=64,
                filter_size=3,
                stride=1,
                act="relu",
                name="conv2",
            )
            self.conv1_3 = ConvBNLayer(
                num_channels=64,
                num_filters=128,
                filter_size=3,
                stride=1,
                act="relu",
                name="conv3",
            )

        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    "bb_%d_%d" % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0
                        else num_filters[block] * int(64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        reduction_ratio=self.reduction_ratio,
                        shortcut=shortcut,
                        if_first=block == 0,
                        name=str(n) + "_" + str(i + 1),
                    ),
                )
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc6_weights"),
            bias_attr=ParamAttr(name="fc6_offset"),
        )

    def forward(self, inputs):
        if self.layers < 152:
            y = self.conv(inputs)
        else:
            y = self.conv1_1(inputs)
            y = self.conv1_2(y)
            y = self.conv1_3(y)
        y = self.pool2d_max(y)

        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def SE_ResNeXt50_32x4d(**args):
    model = ResNeXt(layers=50, cardinality=32, **args)
    return model


def SE_ResNeXt101_32x4d(**args):
    model = ResNeXt(layers=101, cardinality=32, **args)
    return model


def SE_ResNeXt152_64x4d(**args):
    model = ResNeXt(layers=152, cardinality=64, **args)
    return model


def test_SE_ResNeXt50_32x4d(test_case):
    load_paddle_module_and_check(
        test_case, SE_ResNeXt50_32x4d, input_size=(1, 3, 224, 224), train_flag=False,
    )
