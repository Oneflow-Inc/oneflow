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
import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, AdaptiveAvgPool2D, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform, KaimingNormal

__all__ = ["GhostNet_x0_5", "GhostNet_x1_0", "GhostNet_x1_3"]

from oneflow.python.test.onnx.load.util import load_paddle_module_and_check


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        act="relu",
        name=None,
    ):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False,
        )
        bn_name = name + "_bn"

        self._batch_norm = BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale", regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(name=bn_name + "_offset", regularizer=L2Decay(0.0)),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
        )

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SEBlock(nn.Layer):
    def __init__(self, num_channels, reduction_ratio=4, name=None):
        super(SEBlock, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        med_ch = num_channels // reduction_ratio
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_1_weights"
            ),
            bias_attr=ParamAttr(name=name + "_1_offset"),
        )
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_channels,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_2_weights"
            ),
            bias_attr=ParamAttr(name=name + "_2_offset"),
        )

    def forward(self, inputs):
        pool = self.pool2d_gap(inputs)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = paddle.clip(x=excitation, min=0, max=1)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = paddle.multiply(inputs, excitation)
        return out


class GhostModule(nn.Layer):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True,
        name=None,
    ):
        super(GhostModule, self).__init__()
        init_channels = int(math.ceil(output_channels / ratio))
        new_channels = int(init_channels * (ratio - 1))
        self.primary_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
            act="relu" if relu else None,
            name=name + "_primary_conv",
        )
        self.cheap_operation = ConvBNLayer(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=1,
            groups=init_channels,
            act="relu" if relu else None,
            name=name + "_cheap_operation",
        )

    def forward(self, inputs):
        x = self.primary_conv(inputs)
        y = self.cheap_operation(x)
        out = paddle.concat([x, y], axis=1)
        return out


class GhostBottleneck(nn.Layer):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        output_channels,
        kernel_size,
        stride,
        use_se,
        name=None,
    ):
        super(GhostBottleneck, self).__init__()
        self._stride = stride
        self._use_se = use_se
        self._num_channels = in_channels
        self._output_channels = output_channels
        self.ghost_module_1 = GhostModule(
            in_channels=in_channels,
            output_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            name=name + "_ghost_module_1",
        )
        if stride == 2:
            self.depthwise_conv = ConvBNLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
                act=None,
                name=name
                + "_depthwise_depthwise",  # looks strange due to an old typo, will be fixed later.
            )
        if use_se:
            self.se_block = SEBlock(num_channels=hidden_dim, name=name + "_se")
        self.ghost_module_2 = GhostModule(
            in_channels=hidden_dim,
            output_channels=output_channels,
            kernel_size=1,
            relu=False,
            name=name + "_ghost_module_2",
        )
        if stride != 1 or in_channels != output_channels:
            self.shortcut_depthwise = ConvBNLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=in_channels,
                act=None,
                name=name
                + "_shortcut_depthwise_depthwise",  # looks strange due to an old typo, will be fixed later.
            )
            self.shortcut_conv = ConvBNLayer(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                act=None,
                name=name + "_shortcut_conv",
            )

    def forward(self, inputs):
        x = self.ghost_module_1(inputs)
        if self._stride == 2:
            x = self.depthwise_conv(x)
        if self._use_se:
            x = self.se_block(x)
        x = self.ghost_module_2(x)
        if self._stride == 1 and self._num_channels == self._output_channels:
            shortcut = inputs
        else:
            shortcut = self.shortcut_depthwise(inputs)
            shortcut = self.shortcut_conv(shortcut)
        return paddle.add(x=x, y=shortcut)


class GhostNet(nn.Layer):
    def __init__(self, scale, class_dim=1000):
        super(GhostNet, self).__init__()
        self.cfgs = [
            # k, t, c, SE, s
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
        ]
        self.scale = scale
        output_channels = int(self._make_divisible(16 * self.scale, 4))
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            groups=1,
            act="relu",
            name="conv1",
        )
        # build inverted residual blocks
        idx = 0
        self.ghost_bottleneck_list = []
        for k, exp_size, c, use_se, s in self.cfgs:
            in_channels = output_channels
            output_channels = int(self._make_divisible(c * self.scale, 4))
            hidden_dim = int(self._make_divisible(exp_size * self.scale, 4))
            ghost_bottleneck = self.add_sublayer(
                name="_ghostbottleneck_" + str(idx),
                sublayer=GhostBottleneck(
                    in_channels=in_channels,
                    hidden_dim=hidden_dim,
                    output_channels=output_channels,
                    kernel_size=k,
                    stride=s,
                    use_se=use_se,
                    name="_ghostbottleneck_" + str(idx),
                ),
            )
            self.ghost_bottleneck_list.append(ghost_bottleneck)
            idx += 1
        # build last several layers
        in_channels = output_channels
        output_channels = int(self._make_divisible(exp_size * self.scale, 4))
        self.conv_last = ConvBNLayer(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act="relu",
            name="conv_last",
        )
        self.pool2d_gap = AdaptiveAvgPool2D(1)
        in_channels = output_channels
        self._fc0_output_channels = 1280
        self.fc_0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=self._fc0_output_channels,
            kernel_size=1,
            stride=1,
            act="relu",
            name="fc_0",
        )
        self.dropout = nn.Dropout(p=0.2)
        stdv = 1.0 / math.sqrt(self._fc0_output_channels * 1.0)
        self.fc_1 = Linear(
            self._fc0_output_channels,
            class_dim,
            weight_attr=ParamAttr(
                name="fc_1_weights", initializer=Uniform(-stdv, stdv)
            ),
            bias_attr=ParamAttr(name="fc_1_offset"),
        )

    def forward(self, inputs):
        x = self.conv1(inputs)
        for ghost_bottleneck in self.ghost_bottleneck_list:
            x = ghost_bottleneck(x)
        x = self.conv_last(x)
        x = self.pool2d_gap(x)
        x = self.fc_0(x)
        x = self.dropout(x)
        x = paddle.reshape(x, shape=[-1, self._fc0_output_channels])
        x = self.fc_1(x)
        return x

    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def GhostNet_x0_5(**args):
    model = GhostNet(scale=0.5)
    return model


def GhostNet_x1_0(**args):
    model = GhostNet(scale=1.0)
    return model


def GhostNet_x1_3(**args):
    model = GhostNet(scale=1.3)
    return model


def test_GhostNet_x0_5(test_case):
    load_paddle_module_and_check(
        test_case, GhostNet_x0_5, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_GhostNet_x1_0(test_case):
    load_paddle_module_and_check(
        test_case, GhostNet_x1_0, input_size=(1, 3, 224, 224), train_flag=False,
    )


def test_GhostNet_x1_3(test_case):
    load_paddle_module_and_check(
        test_case, GhostNet_x1_3, input_size=(1, 3, 224, 224), train_flag=False,
    )
