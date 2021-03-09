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
import oneflow as flow
import oneflow.typing as tp
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from util import convert_to_onnx_and_check

g_trainable = False


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):

    if data_format == "NCHW":
        weight_shape = (
            int(filters),
            int(input.shape[1] / group_num),
            int(kernel_size[0]),
            int(kernel_size[0]),
        )
    elif data_format == "NHWC":
        weight_shape = (
            int(filters),
            int(kernel_size[0]),
            int(kernel_size[0]),
            int(input.shape[3] / group_num),
        )
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        group_num,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            model_name="bias",
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _batch_norm(
    inputs, axis, momentum, epsilon, center=True, scale=True, trainable=True, name=None
):

    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        trainable=trainable,
        training=trainable,
        name=name,
    )


def _relu6(data, prefix):
    return flow.clip_by_value(data, 0, 6, name="%s-relu6" % prefix)


def mobilenet_unit(
    data,
    num_filter=1,
    kernel=(1, 1),
    stride=(1, 1),
    pad=(0, 0),
    num_group=1,
    data_format="NCHW",
    if_act=True,
    use_bias=False,
    prefix="",
):
    conv = _conv2d_layer(
        name=prefix,
        input=data,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        data_format=data_format,
        dilation_rate=1,
        activation=None,
        use_bias=use_bias,
    )
    if data_format == "NCHW":
        axis = 1
    elif data_format == "NHWC":
        axis = 3
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')

    bn = _batch_norm(
        conv,
        axis=axis,
        momentum=0.97,
        epsilon=1e-3,
        name="%s-BatchNorm" % prefix,
        trainable=g_trainable,
    )
    if if_act:
        act = _relu6(bn, prefix)
        return act
    else:
        return bn


def conv(
    data,
    num_filter=1,
    kernel=(1, 1),
    stride=(1, 1),
    pad=(0, 0),
    num_group=1,
    data_format="NCHW",
    use_bias=False,
    prefix="",
):
    # return _conv2d_layer(name='%s-conv2d'%prefix, input=data, filters=num_filter, kernel_size=kernel, strides=stride, padding=pad, group_num=num_group, dilation_rate=1, activation=None, use_bias=use_bias)
    return _conv2d_layer(
        name=prefix,
        input=data,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        data_format=data_format,
        dilation_rate=1,
        activation=None,
        use_bias=use_bias,
    )


def shortcut(data_in, data_residual, prefix):
    out = flow.math.add(data_in, data_residual)
    return out


def inverted_residual_unit(
    data,
    num_in_filter,
    num_filter,
    ifshortcut,
    stride,
    kernel,
    pad,
    expansion_factor,
    prefix,
    data_format="NCHW",
    has_expand=1,
):
    num_expfilter = int(round(num_in_filter * expansion_factor))
    if has_expand:
        channel_expand = mobilenet_unit(
            data=data,
            num_filter=num_expfilter,
            kernel=(1, 1),
            stride=(1, 1),
            pad="valid",
            num_group=1,
            data_format=data_format,
            if_act=True,
            prefix="%s-expand" % prefix,
        )
    else:
        channel_expand = data
    bottleneck_conv = mobilenet_unit(
        data=channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        data_format=data_format,
        if_act=True,
        prefix="%s-depthwise" % prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad="valid",
        num_group=1,
        data_format=data_format,
        if_act=False,
        prefix="%s-project" % prefix,
    )

    if ifshortcut:
        out = shortcut(data_in=data, data_residual=linear_out, prefix=prefix,)
        return out
    else:
        return linear_out


MNETV2_CONFIGS_MAP = {
    (224, 224): {
        "firstconv_filter_num": 32,
        # t, c, s
        "bottleneck_params_list": [
            (1, 16, 1, False),
            (6, 24, 2, False),
            (6, 24, 1, True),
            (6, 32, 2, False),
            (6, 32, 1, True),
            (6, 32, 1, True),
            (6, 64, 2, False),
            (6, 64, 1, True),
            (6, 64, 1, True),
            (6, 64, 1, True),
            (6, 96, 1, False),
            (6, 96, 1, True),
            (6, 96, 1, True),
            (6, 160, 2, False),
            (6, 160, 1, True),
            (6, 160, 1, True),
            (6, 320, 1, False),
        ],
        "filter_num_before_gp": 1280,
    }
}


class MobileNetV2(object):
    def __init__(self, data_wh, multiplier, **kargs):
        super(MobileNetV2, self).__init__()
        self.data_wh = data_wh
        self.multiplier = multiplier
        if self.data_wh in MNETV2_CONFIGS_MAP:
            self.config_map = MNETV2_CONFIGS_MAP[self.data_wh]
        else:
            self.config_map = MNETV2_CONFIGS_MAP[(224, 224)]

    def build_network(
        self, input_data, data_format, class_num=1000, prefix="", **configs
    ):
        self.config_map.update(configs)

        # input_data = flow.math.multiply(input_data, 2.0 / 255.0)
        # input_data = flow.math.add(input_data, -1)

        if data_format == "NCHW":
            input_data = flow.transpose(input_data, name="transpose", perm=[0, 3, 1, 2])
        first_c = int(round(self.config_map["firstconv_filter_num"] * self.multiplier))
        first_layer = mobilenet_unit(
            data=input_data,
            num_filter=first_c,
            kernel=(3, 3),
            stride=(2, 2),
            pad="same",
            data_format=data_format,
            if_act=True,
            prefix=prefix + "-Conv",
        )

        last_bottleneck_layer = first_layer
        in_c = first_c
        for i, layer_setting in enumerate(self.config_map["bottleneck_params_list"]):
            t, c, s, sc = layer_setting
            if i == 0:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c * self.multiplier)),
                    ifshortcut=sc,
                    stride=(s, s),
                    kernel=(3, 3),
                    pad="same",
                    expansion_factor=t,
                    prefix=prefix + "-expanded_conv",
                    data_format=data_format,
                    has_expand=0,
                )
                in_c = int(round(c * self.multiplier))
            else:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c * self.multiplier)),
                    ifshortcut=sc,
                    stride=(s, s),
                    kernel=(3, 3),
                    pad="same",
                    expansion_factor=t,
                    data_format=data_format,
                    prefix=prefix + "-expanded_conv_%d" % i,
                )
                in_c = int(round(c * self.multiplier))

        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            # num_filter=int(1280 * self.multiplier) if self.multiplier > 1.0 else 1280,
            # gr to confirm
            num_filter=int(256 * self.multiplier) if self.multiplier > 1.0 else 256,
            kernel=(1, 1),
            stride=(1, 1),
            pad="valid",
            data_format=data_format,
            if_act=True,
            prefix=prefix + "-Conv_1",
        )
        base_only = True
        if base_only:
            return last_fm
        else:
            raise NotImplementedError

    def __call__(
        self, input_data, class_num=1000, prefix="", layer_out=None, **configs
    ):
        sym = self.build_network(
            input_data, class_num=class_num, prefix=prefix, **configs
        )
        if layer_out is None:
            return sym

        internals = sym.get_internals()
        if type(layer_out) is list or type(layer_out) is tuple:
            layers_out = [
                internals[layer_nm.strip() + "_output"] for layer_nm in layer_out
            ]
            return layers_out
        else:
            layer_out = internals[layer_out.strip() + "_output"]
            return layer_out


def Mobilenet(
    input_data, data_format="NCHW", num_classes=1000, multiplier=1.0, prefix=""
):
    mobilenetgen = MobileNetV2((224, 224), multiplier=multiplier)
    layer_out = mobilenetgen(
        input_data,
        data_format=data_format,
        class_num=num_classes,
        prefix=prefix + "-MobilenetV2",
        layer_out=None,
    )
    return layer_out


def test_mobilenetv2(test_case):
    @flow.global_function()
    def mobilenetv2(x: tp.Numpy.Placeholder((1, 224, 224, 3))):
        return Mobilenet(x)

    convert_to_onnx_and_check(mobilenetv2)
