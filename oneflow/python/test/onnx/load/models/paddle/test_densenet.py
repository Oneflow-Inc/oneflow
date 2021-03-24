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

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

__all__ = [
    "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "DenseNet264"
]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class BNACConvLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        pad=0,
        groups=1,
        act="relu",
        name=None,
    ):
        super(BNACConvLayer, self).__init__()

        self._batch_norm = BatchNorm(
            num_channels,
            act=act,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance",
        )

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
        )

    def forward(self, input):
        y = self._batch_norm(input)
        y = self._conv(y)
        return y


class DenseLayer(nn.Layer):
    def __init__(self, num_channels, growth_rate, bn_size, dropout, name=None):
        super(DenseLayer, self).__init__()
        self.dropout = dropout

        self.bn_ac_func1 = BNACConvLayer(
            num_channels=num_channels,
            num_filters=bn_size * growth_rate,
            filter_size=1,
            pad=0,
            stride=1,
            name=name + "_x1",
        )

        self.bn_ac_func2 = BNACConvLayer(
            num_channels=bn_size * growth_rate,
            num_filters=growth_rate,
            filter_size=3,
            pad=1,
            stride=1,
            name=name + "_x2",
        )

        if dropout:
            self.dropout_func = Dropout(p=dropout, mode="downscale_in_infer")

    def forward(self, input):
        conv = self.bn_ac_func1(input)
        conv = self.bn_ac_func2(conv)
        if self.dropout:
            conv = self.dropout_func(conv)
        conv = paddle.concat([input, conv], axis=1)
        return conv


class DenseBlock(nn.Layer):
    def __init__(
        self, num_channels, num_layers, bn_size, growth_rate, dropout, name=None
    ):
        super(DenseBlock, self).__init__()
        self.dropout = dropout

        self.dense_layer_func = []

        pre_channel = num_channels
        for layer in range(num_layers):
            self.dense_layer_func.append(
                self.add_sublayer(
                    "{}_{}".format(name, layer + 1),
                    DenseLayer(
                        num_channels=pre_channel,
                        growth_rate=growth_rate,
                        bn_size=bn_size,
                        dropout=dropout,
                        name=name + "_" + str(layer + 1),
                    ),
                )
            )
            pre_channel = pre_channel + growth_rate

    def forward(self, input):
        conv = input
        for func in self.dense_layer_func:
            conv = func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self, num_channels, num_output_features, name=None):
        super(TransitionLayer, self).__init__()

        self.conv_ac_func = BNACConvLayer(
            num_channels=num_channels,
            num_filters=num_output_features,
            filter_size=1,
            pad=0,
            stride=1,
            name=name,
        )

        self.pool2d_avg = AvgPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        y = self.conv_ac_func(input)
        y = self.pool2d_avg(y)
        return y


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        pad=0,
        groups=1,
        act="relu",
        name=None,
    ):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
        )
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance",
        )

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


class DenseNet(nn.Layer):
    def __init__(self, layers=60, bn_size=4, dropout=0, class_dim=1000):
        super(DenseNet, self).__init__()

        supported_layers = [121, 161, 169, 201, 264]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )
        densenet_spec = {
            121: (64, 32, [6, 12, 24, 16]),
            161: (96, 48, [6, 12, 36, 24]),
            169: (64, 32, [6, 12, 32, 32]),
            201: (64, 32, [6, 12, 48, 32]),
            264: (64, 32, [6, 12, 64, 48]),
        }
        num_init_features, growth_rate, block_config = densenet_spec[layers]

        self.conv1_func = ConvBNLayer(
            num_channels=3,
            num_filters=num_init_features,
            filter_size=7,
            stride=2,
            pad=3,
            act="relu",
            name="conv1",
        )

        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_config = block_config

        self.dense_block_func_list = []
        self.transition_func_list = []
        pre_num_channels = num_init_features
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.dense_block_func_list.append(
                self.add_sublayer(
                    "db_conv_{}".format(i + 2),
                    DenseBlock(
                        num_channels=pre_num_channels,
                        num_layers=num_layers,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        dropout=dropout,
                        name="conv" + str(i + 2),
                    ),
                )
            )

            num_features = num_features + num_layers * growth_rate
            pre_num_channels = num_features

            if i != len(block_config) - 1:
                self.transition_func_list.append(
                    self.add_sublayer(
                        "tr_conv{}_blk".format(i + 2),
                        TransitionLayer(
                            num_channels=pre_num_channels,
                            num_output_features=num_features // 2,
                            name="conv" + str(i + 2) + "_blk",
                        ),
                    )
                )
                pre_num_channels = num_features // 2
                num_features = num_features // 2

        self.batch_norm = BatchNorm(
            num_features,
            act="relu",
            param_attr=ParamAttr(name="conv5_blk_bn_scale"),
            bias_attr=ParamAttr(name="conv5_blk_bn_offset"),
            moving_mean_name="conv5_blk_bn_mean",
            moving_variance_name="conv5_blk_bn_variance",
        )

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        stdv = 1.0 / math.sqrt(num_features * 1.0)

        self.out = Linear(
            num_features,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"),
        )

    def forward(self, input):
        conv = self.conv1_func(input)
        conv = self.pool2d_max(conv)

        for i, num_layers in enumerate(self.block_config):
            conv = self.dense_block_func_list[i](conv)
            if i != len(self.block_config) - 1:
                conv = self.transition_func_list[i](conv)

        conv = self.batch_norm(conv)
        y = self.pool2d_avg(conv)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        y = self.out(y)
        return y


def DenseNet121(**args):
    model = DenseNet(layers=121, **args)
    return model


def DenseNet161(**args):
    model = DenseNet(layers=161, **args)
    return model


def DenseNet169(**args):
    model = DenseNet(layers=169, **args)
    return model


def DenseNet201(**args):
    model = DenseNet(layers=201, **args)
    return model


def DenseNet264(**args):
    model = DenseNet(layers=264, **args)
    return model


def test_densenet121(test_case):
    load_paddle_module_and_check(
        test_case, DenseNet121, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_densenet161(test_case):
    load_paddle_module_and_check(
        test_case, DenseNet161, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_densenet169(test_case):
    load_paddle_module_and_check(
        test_case, DenseNet169, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_densenet201(test_case):
    load_paddle_module_and_check(
        test_case, DenseNet201, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_densenet264(test_case):
    load_paddle_module_and_check(
        test_case, DenseNet264, input_size=(1, 3, 224, 224), train_flag=False,
    )
