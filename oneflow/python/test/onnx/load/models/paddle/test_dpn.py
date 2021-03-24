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
import sys
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

__all__ = [
    "DPN",
    "DPN68",
    "DPN92",
    "DPN98",
    "DPN107",
    "DPN131",
]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


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
        self.num_channels = num_channels

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


class DualPathFactory(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_1x1_a,
        num_3x3_b,
        num_1x1_c,
        inc,
        G,
        _type="normal",
        name=None,
    ):
        super(DualPathFactory, self).__init__()

        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.name = name

        kw = 3
        kh = 3
        pw = (kw - 1) // 2
        ph = (kh - 1) // 2

        # type
        if _type == "proj":
            key_stride = 1
            self.has_proj = True
        elif _type == "down":
            key_stride = 2
            self.has_proj = True
        elif _type == "normal":
            key_stride = 1
            self.has_proj = False
        else:
            print("not implemented now!!!")
            sys.exit(1)

        data_in_ch = (
            sum(num_channels) if isinstance(num_channels, list) else num_channels
        )

        if self.has_proj:
            self.c1x1_w_func = BNACConvLayer(
                num_channels=data_in_ch,
                num_filters=num_1x1_c + 2 * inc,
                filter_size=(1, 1),
                pad=(0, 0),
                stride=(key_stride, key_stride),
                name=name + "_match",
            )

        self.c1x1_a_func = BNACConvLayer(
            num_channels=data_in_ch,
            num_filters=num_1x1_a,
            filter_size=(1, 1),
            pad=(0, 0),
            name=name + "_conv1",
        )

        self.c3x3_b_func = BNACConvLayer(
            num_channels=num_1x1_a,
            num_filters=num_3x3_b,
            filter_size=(kw, kh),
            pad=(pw, ph),
            stride=(key_stride, key_stride),
            groups=G,
            name=name + "_conv2",
        )

        self.c1x1_c_func = BNACConvLayer(
            num_channels=num_3x3_b,
            num_filters=num_1x1_c + inc,
            filter_size=(1, 1),
            pad=(0, 0),
            name=name + "_conv3",
        )

    def forward(self, input):
        # PROJ
        if isinstance(input, list):
            data_in = paddle.concat([input[0], input[1]], axis=1)
        else:
            data_in = input

        if self.has_proj:
            c1x1_w = self.c1x1_w_func(data_in)
            data_o1, data_o2 = paddle.split(
                c1x1_w, num_or_sections=[self.num_1x1_c, 2 * self.inc], axis=1
            )
        else:
            data_o1 = input[0]
            data_o2 = input[1]

        c1x1_a = self.c1x1_a_func(data_in)
        c3x3_b = self.c3x3_b_func(c1x1_a)
        c1x1_c = self.c1x1_c_func(c3x3_b)

        c1x1_c1, c1x1_c2 = paddle.split(
            c1x1_c, num_or_sections=[self.num_1x1_c, self.inc], axis=1
        )

        # OUTPUTS
        summ = paddle.add(x=data_o1, y=c1x1_c1)
        dense = paddle.concat([data_o2, c1x1_c2], axis=1)
        # tensor, channels
        return [summ, dense]


class DPN(nn.Layer):
    def __init__(self, layers=68, class_dim=1000):
        super(DPN, self).__init__()

        self._class_dim = class_dim

        args = self.get_net_args(layers)
        bws = args["bw"]
        inc_sec = args["inc_sec"]
        rs = args["r"]
        k_r = args["k_r"]
        k_sec = args["k_sec"]
        G = args["G"]
        init_num_filter = args["init_num_filter"]
        init_filter_size = args["init_filter_size"]
        init_padding = args["init_padding"]

        self.k_sec = k_sec

        self.conv1_x_1_func = ConvBNLayer(
            num_channels=3,
            num_filters=init_num_filter,
            filter_size=init_filter_size,
            stride=2,
            pad=init_padding,
            act="relu",
            name="conv1",
        )

        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        num_channel_dpn = init_num_filter

        self.dpn_func_list = []
        # conv2 - conv5
        match_list, num = [], 0
        for gc in range(4):
            bw = bws[gc]
            inc = inc_sec[gc]
            R = (k_r * bw) // rs[gc]
            if gc == 0:
                _type1 = "proj"
                _type2 = "normal"
                match = 1
            else:
                _type1 = "down"
                _type2 = "normal"
                match = match + k_sec[gc - 1]
            match_list.append(match)
            self.dpn_func_list.append(
                self.add_sublayer(
                    "dpn{}".format(match),
                    DualPathFactory(
                        num_channels=num_channel_dpn,
                        num_1x1_a=R,
                        num_3x3_b=R,
                        num_1x1_c=bw,
                        inc=inc,
                        G=G,
                        _type=_type1,
                        name="dpn" + str(match),
                    ),
                )
            )
            num_channel_dpn = [bw, 3 * inc]

            for i_ly in range(2, k_sec[gc] + 1):
                num += 1
                if num in match_list:
                    num += 1
                self.dpn_func_list.append(
                    self.add_sublayer(
                        "dpn{}".format(num),
                        DualPathFactory(
                            num_channels=num_channel_dpn,
                            num_1x1_a=R,
                            num_3x3_b=R,
                            num_1x1_c=bw,
                            inc=inc,
                            G=G,
                            _type=_type2,
                            name="dpn" + str(num),
                        ),
                    )
                )

                num_channel_dpn = [num_channel_dpn[0], num_channel_dpn[1] + inc]

        out_channel = sum(num_channel_dpn)

        self.conv5_x_x_bn = BatchNorm(
            num_channels=sum(num_channel_dpn),
            act="relu",
            param_attr=ParamAttr(name="final_concat_bn_scale"),
            bias_attr=ParamAttr("final_concat_bn_offset"),
            moving_mean_name="final_concat_bn_mean",
            moving_variance_name="final_concat_bn_variance",
        )

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        stdv = 0.01

        self.out = Linear(
            out_channel,
            class_dim,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"),
        )

    def forward(self, input):
        conv1_x_1 = self.conv1_x_1_func(input)
        convX_x_x = self.pool2d_max(conv1_x_1)

        dpn_idx = 0
        for gc in range(4):
            convX_x_x = self.dpn_func_list[dpn_idx](convX_x_x)
            dpn_idx += 1
            for i_ly in range(2, self.k_sec[gc] + 1):
                convX_x_x = self.dpn_func_list[dpn_idx](convX_x_x)
                dpn_idx += 1

        conv5_x_x = paddle.concat(convX_x_x, axis=1)
        conv5_x_x = self.conv5_x_x_bn(conv5_x_x)

        y = self.pool2d_avg(conv5_x_x)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        y = self.out(y)
        return y

    def get_net_args(self, layers):
        if layers == 68:
            k_r = 128
            G = 32
            k_sec = [3, 4, 12, 3]
            inc_sec = [16, 32, 32, 64]
            bw = [64, 128, 256, 512]
            r = [64, 64, 64, 64]
            init_num_filter = 10
            init_filter_size = 3
            init_padding = 1
        elif layers == 92:
            k_r = 96
            G = 32
            k_sec = [3, 4, 20, 3]
            inc_sec = [16, 32, 24, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 64
            init_filter_size = 7
            init_padding = 3
        elif layers == 98:
            k_r = 160
            G = 40
            k_sec = [3, 6, 20, 3]
            inc_sec = [16, 32, 32, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 96
            init_filter_size = 7
            init_padding = 3
        elif layers == 107:
            k_r = 200
            G = 50
            k_sec = [4, 8, 20, 3]
            inc_sec = [20, 64, 64, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 128
            init_filter_size = 7
            init_padding = 3
        elif layers == 131:
            k_r = 160
            G = 40
            k_sec = [4, 8, 28, 3]
            inc_sec = [16, 32, 32, 128]
            bw = [256, 512, 1024, 2048]
            r = [256, 256, 256, 256]
            init_num_filter = 128
            init_filter_size = 7
            init_padding = 3
        else:
            raise NotImplementedError
        net_arg = {
            "k_r": k_r,
            "G": G,
            "k_sec": k_sec,
            "inc_sec": inc_sec,
            "bw": bw,
            "r": r,
        }
        net_arg["init_num_filter"] = init_num_filter
        net_arg["init_filter_size"] = init_filter_size
        net_arg["init_padding"] = init_padding

        return net_arg


def DPN68(**args):
    model = DPN(layers=68, **args)
    return model


def DPN92(**args):
    model = DPN(layers=92, **args)
    return model


def DPN98(**args):
    model = DPN(layers=98, **args)
    return model


def DPN107(**args):
    model = DPN(layers=107, **args)
    return model


def DPN131(**args):
    model = DPN(layers=131, **args)
    return model


def test_dpn68(test_case):
    load_paddle_module_and_check(
        test_case, DPN68, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_dpn92(test_case):
    load_paddle_module_and_check(
        test_case, DPN92, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_dpn98(test_case):
    load_paddle_module_and_check(
        test_case, DPN98, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_dpn107(test_case):
    load_paddle_module_and_check(
        test_case, DPN107, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_dpn131(test_case):
    load_paddle_module_and_check(
        test_case, DPN131, input_size=(1, 3, 224, 224), train_flag=False,
    )
