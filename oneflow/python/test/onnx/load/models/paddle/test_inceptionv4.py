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

__all__ = ["InceptionV4"]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        padding=0,
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
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
        )
        bn_name = name + "_bn"
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


class InceptionStem(nn.Layer):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self._conv_1 = ConvBNLayer(3, 32, 3, stride=2, act="relu", name="conv1_3x3_s2")
        self._conv_2 = ConvBNLayer(32, 32, 3, act="relu", name="conv2_3x3_s1")
        self._conv_3 = ConvBNLayer(
            32, 64, 3, padding=1, act="relu", name="conv3_3x3_s1"
        )
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self._conv2 = ConvBNLayer(
            64, 96, 3, stride=2, act="relu", name="inception_stem1_3x3_s2"
        )
        self._conv1_1 = ConvBNLayer(
            160, 64, 1, act="relu", name="inception_stem2_3x3_reduce"
        )
        self._conv1_2 = ConvBNLayer(64, 96, 3, act="relu", name="inception_stem2_3x3")
        self._conv2_1 = ConvBNLayer(
            160, 64, 1, act="relu", name="inception_stem2_1x7_reduce"
        )
        self._conv2_2 = ConvBNLayer(
            64, 64, (7, 1), padding=(3, 0), act="relu", name="inception_stem2_1x7"
        )
        self._conv2_3 = ConvBNLayer(
            64, 64, (1, 7), padding=(0, 3), act="relu", name="inception_stem2_7x1"
        )
        self._conv2_4 = ConvBNLayer(64, 96, 3, act="relu", name="inception_stem2_3x3_2")
        self._conv3 = ConvBNLayer(
            192, 192, 3, stride=2, act="relu", name="inception_stem3_3x3_s2"
        )

    def forward(self, inputs):
        conv = self._conv_1(inputs)
        conv = self._conv_2(conv)
        conv = self._conv_3(conv)

        pool1 = self._pool(conv)
        conv2 = self._conv2(conv)
        concat = paddle.concat([pool1, conv2], axis=1)

        conv1 = self._conv1_1(concat)
        conv1 = self._conv1_2(conv1)

        conv2 = self._conv2_1(concat)
        conv2 = self._conv2_2(conv2)
        conv2 = self._conv2_3(conv2)
        conv2 = self._conv2_4(conv2)

        concat = paddle.concat([conv1, conv2], axis=1)

        conv1 = self._conv3(concat)
        pool1 = self._pool(concat)

        concat = paddle.concat([conv1, pool1], axis=1)
        return concat


class InceptionA(nn.Layer):
    def __init__(self, name):
        super(InceptionA, self).__init__()
        self._pool = AvgPool2D(kernel_size=3, stride=1, padding=1)
        self._conv1 = ConvBNLayer(
            384, 96, 1, act="relu", name="inception_a" + name + "_1x1"
        )
        self._conv2 = ConvBNLayer(
            384, 96, 1, act="relu", name="inception_a" + name + "_1x1_2"
        )
        self._conv3_1 = ConvBNLayer(
            384, 64, 1, act="relu", name="inception_a" + name + "_3x3_reduce"
        )
        self._conv3_2 = ConvBNLayer(
            64, 96, 3, padding=1, act="relu", name="inception_a" + name + "_3x3"
        )
        self._conv4_1 = ConvBNLayer(
            384, 64, 1, act="relu", name="inception_a" + name + "_3x3_2_reduce"
        )
        self._conv4_2 = ConvBNLayer(
            64, 96, 3, padding=1, act="relu", name="inception_a" + name + "_3x3_2"
        )
        self._conv4_3 = ConvBNLayer(
            96, 96, 3, padding=1, act="relu", name="inception_a" + name + "_3x3_3"
        )

    def forward(self, inputs):
        pool1 = self._pool(inputs)
        conv1 = self._conv1(pool1)

        conv2 = self._conv2(inputs)

        conv3 = self._conv3_1(inputs)
        conv3 = self._conv3_2(conv3)

        conv4 = self._conv4_1(inputs)
        conv4 = self._conv4_2(conv4)
        conv4 = self._conv4_3(conv4)

        concat = paddle.concat([conv1, conv2, conv3, conv4], axis=1)
        return concat


class ReductionA(nn.Layer):
    def __init__(self):
        super(ReductionA, self).__init__()
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self._conv2 = ConvBNLayer(
            384, 384, 3, stride=2, act="relu", name="reduction_a_3x3"
        )
        self._conv3_1 = ConvBNLayer(
            384, 192, 1, act="relu", name="reduction_a_3x3_2_reduce"
        )
        self._conv3_2 = ConvBNLayer(
            192, 224, 3, padding=1, act="relu", name="reduction_a_3x3_2"
        )
        self._conv3_3 = ConvBNLayer(
            224, 256, 3, stride=2, act="relu", name="reduction_a_3x3_3"
        )

    def forward(self, inputs):
        pool1 = self._pool(inputs)
        conv2 = self._conv2(inputs)
        conv3 = self._conv3_1(inputs)
        conv3 = self._conv3_2(conv3)
        conv3 = self._conv3_3(conv3)
        concat = paddle.concat([pool1, conv2, conv3], axis=1)
        return concat


class InceptionB(nn.Layer):
    def __init__(self, name=None):
        super(InceptionB, self).__init__()
        self._pool = AvgPool2D(kernel_size=3, stride=1, padding=1)
        self._conv1 = ConvBNLayer(
            1024, 128, 1, act="relu", name="inception_b" + name + "_1x1"
        )
        self._conv2 = ConvBNLayer(
            1024, 384, 1, act="relu", name="inception_b" + name + "_1x1_2"
        )
        self._conv3_1 = ConvBNLayer(
            1024, 192, 1, act="relu", name="inception_b" + name + "_1x7_reduce"
        )
        self._conv3_2 = ConvBNLayer(
            192,
            224,
            (1, 7),
            padding=(0, 3),
            act="relu",
            name="inception_b" + name + "_1x7",
        )
        self._conv3_3 = ConvBNLayer(
            224,
            256,
            (7, 1),
            padding=(3, 0),
            act="relu",
            name="inception_b" + name + "_7x1",
        )
        self._conv4_1 = ConvBNLayer(
            1024, 192, 1, act="relu", name="inception_b" + name + "_7x1_2_reduce"
        )
        self._conv4_2 = ConvBNLayer(
            192,
            192,
            (1, 7),
            padding=(0, 3),
            act="relu",
            name="inception_b" + name + "_1x7_2",
        )
        self._conv4_3 = ConvBNLayer(
            192,
            224,
            (7, 1),
            padding=(3, 0),
            act="relu",
            name="inception_b" + name + "_7x1_2",
        )
        self._conv4_4 = ConvBNLayer(
            224,
            224,
            (1, 7),
            padding=(0, 3),
            act="relu",
            name="inception_b" + name + "_1x7_3",
        )
        self._conv4_5 = ConvBNLayer(
            224,
            256,
            (7, 1),
            padding=(3, 0),
            act="relu",
            name="inception_b" + name + "_7x1_3",
        )

    def forward(self, inputs):
        pool1 = self._pool(inputs)
        conv1 = self._conv1(pool1)

        conv2 = self._conv2(inputs)

        conv3 = self._conv3_1(inputs)
        conv3 = self._conv3_2(conv3)
        conv3 = self._conv3_3(conv3)

        conv4 = self._conv4_1(inputs)
        conv4 = self._conv4_2(conv4)
        conv4 = self._conv4_3(conv4)
        conv4 = self._conv4_4(conv4)
        conv4 = self._conv4_5(conv4)

        concat = paddle.concat([conv1, conv2, conv3, conv4], axis=1)
        return concat


class ReductionB(nn.Layer):
    def __init__(self):
        super(ReductionB, self).__init__()
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self._conv2_1 = ConvBNLayer(
            1024, 192, 1, act="relu", name="reduction_b_3x3_reduce"
        )
        self._conv2_2 = ConvBNLayer(
            192, 192, 3, stride=2, act="relu", name="reduction_b_3x3"
        )
        self._conv3_1 = ConvBNLayer(
            1024, 256, 1, act="relu", name="reduction_b_1x7_reduce"
        )
        self._conv3_2 = ConvBNLayer(
            256, 256, (1, 7), padding=(0, 3), act="relu", name="reduction_b_1x7"
        )
        self._conv3_3 = ConvBNLayer(
            256, 320, (7, 1), padding=(3, 0), act="relu", name="reduction_b_7x1"
        )
        self._conv3_4 = ConvBNLayer(
            320, 320, 3, stride=2, act="relu", name="reduction_b_3x3_2"
        )

    def forward(self, inputs):
        pool1 = self._pool(inputs)

        conv2 = self._conv2_1(inputs)
        conv2 = self._conv2_2(conv2)

        conv3 = self._conv3_1(inputs)
        conv3 = self._conv3_2(conv3)
        conv3 = self._conv3_3(conv3)
        conv3 = self._conv3_4(conv3)

        concat = paddle.concat([pool1, conv2, conv3], axis=1)

        return concat


class InceptionC(nn.Layer):
    def __init__(self, name=None):
        super(InceptionC, self).__init__()
        self._pool = AvgPool2D(kernel_size=3, stride=1, padding=1)
        self._conv1 = ConvBNLayer(
            1536, 256, 1, act="relu", name="inception_c" + name + "_1x1"
        )
        self._conv2 = ConvBNLayer(
            1536, 256, 1, act="relu", name="inception_c" + name + "_1x1_2"
        )
        self._conv3_0 = ConvBNLayer(
            1536, 384, 1, act="relu", name="inception_c" + name + "_1x1_3"
        )
        self._conv3_1 = ConvBNLayer(
            384,
            256,
            (1, 3),
            padding=(0, 1),
            act="relu",
            name="inception_c" + name + "_1x3",
        )
        self._conv3_2 = ConvBNLayer(
            384,
            256,
            (3, 1),
            padding=(1, 0),
            act="relu",
            name="inception_c" + name + "_3x1",
        )
        self._conv4_0 = ConvBNLayer(
            1536, 384, 1, act="relu", name="inception_c" + name + "_1x1_4"
        )
        self._conv4_00 = ConvBNLayer(
            384,
            448,
            (1, 3),
            padding=(0, 1),
            act="relu",
            name="inception_c" + name + "_1x3_2",
        )
        self._conv4_000 = ConvBNLayer(
            448,
            512,
            (3, 1),
            padding=(1, 0),
            act="relu",
            name="inception_c" + name + "_3x1_2",
        )
        self._conv4_1 = ConvBNLayer(
            512,
            256,
            (1, 3),
            padding=(0, 1),
            act="relu",
            name="inception_c" + name + "_1x3_3",
        )
        self._conv4_2 = ConvBNLayer(
            512,
            256,
            (3, 1),
            padding=(1, 0),
            act="relu",
            name="inception_c" + name + "_3x1_3",
        )

    def forward(self, inputs):
        pool1 = self._pool(inputs)
        conv1 = self._conv1(pool1)

        conv2 = self._conv2(inputs)

        conv3 = self._conv3_0(inputs)
        conv3_1 = self._conv3_1(conv3)
        conv3_2 = self._conv3_2(conv3)

        conv4 = self._conv4_0(inputs)
        conv4 = self._conv4_00(conv4)
        conv4 = self._conv4_000(conv4)
        conv4_1 = self._conv4_1(conv4)
        conv4_2 = self._conv4_2(conv4)

        concat = paddle.concat(
            [conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2], axis=1
        )

        return concat


class InceptionV4DY(nn.Layer):
    def __init__(self, class_dim=1000):
        super(InceptionV4DY, self).__init__()
        self._inception_stem = InceptionStem()

        self._inceptionA_1 = InceptionA(name="1")
        self._inceptionA_2 = InceptionA(name="2")
        self._inceptionA_3 = InceptionA(name="3")
        self._inceptionA_4 = InceptionA(name="4")
        self._reductionA = ReductionA()

        self._inceptionB_1 = InceptionB(name="1")
        self._inceptionB_2 = InceptionB(name="2")
        self._inceptionB_3 = InceptionB(name="3")
        self._inceptionB_4 = InceptionB(name="4")
        self._inceptionB_5 = InceptionB(name="5")
        self._inceptionB_6 = InceptionB(name="6")
        self._inceptionB_7 = InceptionB(name="7")
        self._reductionB = ReductionB()

        self._inceptionC_1 = InceptionC(name="1")
        self._inceptionC_2 = InceptionC(name="2")
        self._inceptionC_3 = InceptionC(name="3")

        self.avg_pool = AdaptiveAvgPool2D(1)
        self._drop = Dropout(p=0.2, mode="downscale_in_infer")
        stdv = 1.0 / math.sqrt(1536 * 1.0)
        self.out = Linear(
            1536,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="final_fc_weights"
            ),
            bias_attr=ParamAttr(name="final_fc_offset"),
        )

    def forward(self, inputs):
        x = self._inception_stem(inputs)

        x = self._inceptionA_1(x)
        x = self._inceptionA_2(x)
        x = self._inceptionA_3(x)
        x = self._inceptionA_4(x)
        x = self._reductionA(x)

        x = self._inceptionB_1(x)
        x = self._inceptionB_2(x)
        x = self._inceptionB_3(x)
        x = self._inceptionB_4(x)
        x = self._inceptionB_5(x)
        x = self._inceptionB_6(x)
        x = self._inceptionB_7(x)
        x = self._reductionB(x)

        x = self._inceptionC_1(x)
        x = self._inceptionC_2(x)
        x = self._inceptionC_3(x)

        x = self.avg_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self._drop(x)
        x = self.out(x)
        return x


def InceptionV4(**args):
    model = InceptionV4DY(**args)
    return model


def test_InceptionV4(test_case):
    load_paddle_module_and_check(
        test_case, InceptionV4, input_size=(1, 3, 224, 224), train_flag=False,
    )

