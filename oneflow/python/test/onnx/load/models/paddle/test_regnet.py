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
    "RegNetX_200MF",
    "RegNetX_4GF",
    "RegNetX_32GF",
    "RegNetY_200MF",
    "RegNetY_4GF",
    "RegNetY_32GF",
]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts = [
        w != wp or r != rp
        for w, wp, r, rp in zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        padding=0,
        act=None,
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
            weight_attr=ParamAttr(name=name + ".conv2d.output.1.w_0"),
            bias_attr=ParamAttr(name=name + ".conv2d.output.1.b_0"),
        )
        bn_name = name + "_bn"
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + ".output.1.w_0"),
            bias_attr=ParamAttr(bn_name + ".output.1.b_0"),
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
        bm,
        gw,
        se_on,
        se_r,
        shortcut=True,
        name=None,
    ):
        super(BottleneckBlock, self).__init__()

        # Compute the bottleneck width
        w_b = int(round(num_filters * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        self.se_on = se_on
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=w_b,
            filter_size=1,
            padding=0,
            act="relu",
            name=name + "_branch2a",
        )
        self.conv1 = ConvBNLayer(
            num_channels=w_b,
            num_filters=w_b,
            filter_size=3,
            stride=stride,
            padding=1,
            groups=num_gs,
            act="relu",
            name=name + "_branch2b",
        )
        if se_on:
            w_se = int(round(num_channels * se_r))
            self.se_block = SELayer(
                num_channels=w_b,
                num_filters=w_b,
                reduction_ratio=w_se,
                name=name + "_branch2se",
            )
        self.conv2 = ConvBNLayer(
            num_channels=w_b,
            num_filters=num_filters,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                name=name + "_branch1",
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.se_on:
            conv1 = self.se_block(conv1)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
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

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_exc_weights"
            ),
            bias_attr=ParamAttr(name=name + "_exc_offset"),
        )

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = paddle.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.reshape(excitation, shape=[-1, self._num_channels, 1, 1])
        out = input * excitation
        return out


class RegNet(nn.Layer):
    def __init__(
        self, w_a, w_0, w_m, d, group_w, bot_mul, q=8, se_on=False, class_dim=1000
    ):
        super(RegNet, self).__init__()

        # Generate RegNet ws per block
        b_ws, num_s, max_s, ws_cont = generate_regnet(w_a, w_0, w_m, d, q)
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [group_w for _ in range(num_s)]
        bms = [bot_mul for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25
        # Construct the model
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        stem_type = "simple_stem_in"
        stem_w = 32
        block_type = "res_bottleneck_block"

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=stem_w,
            filter_size=3,
            stride=2,
            padding=1,
            act="relu",
            name="stem_conv",
        )

        self.block_list = []
        for block, (d, w_out, stride, bm, gw) in enumerate(stage_params):
            shortcut = False
            for i in range(d):
                num_channels = stem_w if block == i == 0 else in_channels
                # Stride apply to the first block of the stage
                b_stride = stride if i == 0 else 1
                conv_name = "s" + str(block + 1) + "_b" + str(i + 1)  # chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    conv_name,
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=w_out,
                        stride=b_stride,
                        bm=bm,
                        gw=gw,
                        se_on=se_on,
                        se_r=se_r,
                        shortcut=shortcut,
                        name=conv_name,
                    ),
                )
                in_channels = w_out
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.pool2d_avg_channels = w_out

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"),
        )

    def forward(self, inputs):
        y = self.conv(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def RegNetX_200MF(**args):
    model = RegNet(
        w_a=36.44, w_0=24, w_m=2.49, d=13, group_w=8, bot_mul=1.0, q=8, **args
    )
    return model


def RegNetX_4GF(**args):
    model = RegNet(
        w_a=38.65, w_0=96, w_m=2.43, d=23, group_w=40, bot_mul=1.0, q=8, **args
    )
    return model


def RegNetX_32GF(**args):
    model = RegNet(
        w_a=69.86, w_0=320, w_m=2.0, d=23, group_w=168, bot_mul=1.0, q=8, **args
    )
    return model


def RegNetY_200MF(**args):
    model = RegNet(
        w_a=36.44,
        w_0=24,
        w_m=2.49,
        d=13,
        group_w=8,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **args
    )
    return model


def RegNetY_4GF(**args):
    model = RegNet(
        w_a=31.41,
        w_0=96,
        w_m=2.24,
        d=22,
        group_w=64,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **args
    )
    return model


def RegNetY_32GF(**args):
    model = RegNet(
        w_a=115.89,
        w_0=232,
        w_m=2.53,
        d=20,
        group_w=232,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **args
    )
    return model


def test_RegNetX_200MF(test_case):
    load_paddle_module_and_check(
        test_case, RegNetX_200MF, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_RegNetX_4GF(test_case):
    load_paddle_module_and_check(
        test_case, RegNetX_4GF, input_size=(1, 3, 224, 224), train_flag=False,
    )
