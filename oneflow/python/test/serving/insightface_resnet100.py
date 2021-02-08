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
import oneflow.core.operator.op_conf_pb2 as op_conf_util


def _get_initializer():
    return flow.variance_scaling_initializer(2.0, "fan_out", "random_normal", "NCHW")


def _get_regularizer():
    return flow.regularizers.l2(0.0005)


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
    activation=None,
    use_bias=False,
    weight_initializer=_get_initializer(),
    bias_initializer=flow.zeros_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
):
    weight_shape = (
        int(filters),
        int(input.shape[1] / group_num),
        int(kernel_size),
        int(kernel_size),
    )
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    output = flow.nn.conv2d(
        input,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        groups=group_num,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.math.relu(output)
        else:
            raise NotImplementedError

    return output


def _batch_norm(
    inputs,
    epsilon,
    center=True,
    scale=True,
    trainable=True,
    is_training=True,
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.9,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        beta_regularizer=_get_regularizer(),
        gamma_regularizer=_get_regularizer(),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=is_training,
        name=name,
    )


def _prelu(inputs, name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer(),
        shared_axes=[2, 3],
        name=name,
    )


def _avg_pool(inputs, pool_size, strides, padding, name=None):
    return flow.nn.avg_pool2d(
        input=inputs, ksize=pool_size, strides=strides, padding=padding,
    )


def _dropout(input_blob, dropout_prob):
    return flow.nn.dropout(input_blob, rate=dropout_prob)


def Linear(
    input_blob,
    num_filter=1,
    kernel=None,
    stride=None,
    pad="valid",
    num_group=1,
    bn_is_training=True,
    name=None,
    suffix="",
):
    conv = _conv2d_layer(
        name="%s%s_conv2d" % (name, suffix),
        input=input_blob,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn = _batch_norm(
        conv,
        epsilon=0.001,
        is_training=bn_is_training,
        name="%s%s_batchnorm" % (name, suffix),
    )
    return bn


def residual_unit_v3(in_data, num_filter, stride, dim_match, bn_is_training, name):

    suffix = ""
    use_se = 0
    bn1 = _batch_norm(
        in_data,
        epsilon=2e-5,
        is_training=bn_is_training,
        name="%s%s_bn1" % (name, suffix),
    )
    conv1 = _conv2d_layer(
        name="%s%s_conv1" % (name, suffix),
        input=bn1,
        filters=num_filter,
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn2 = _batch_norm(
        conv1,
        epsilon=2e-5,
        is_training=bn_is_training,
        name="%s%s_bn2" % (name, suffix),
    )
    prelu = _prelu(bn2, name="%s%s_relu1" % (name, suffix))
    conv2 = _conv2d_layer(
        name="%s%s_conv2" % (name, suffix),
        input=prelu,
        filters=num_filter,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn3 = _batch_norm(
        conv2,
        epsilon=2e-5,
        is_training=bn_is_training,
        name="%s%s_bn3" % (name, suffix),
    )

    if use_se:
        # se begin
        input_blob = _avg_pool(bn3, pool_size=[7, 7], strides=[1, 1], padding="VALID")
        input_blob = _conv2d_layer(
            name="%s%s_se_conv1" % (name, suffix),
            input=input_blob,
            filters=num_filter // 16,
            kernel_size=1,
            strides=[1, 1],
            padding="valid",
            use_bias=True,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _prelu(input_blob, name="%s%s_se_relu1" % (name, suffix))
        input_blob = _conv2d_layer(
            name="%s%s_se_conv2" % (name, suffix),
            input=input_blob,
            filters=num_filter,
            kernel_size=1,
            strides=[1, 1],
            padding="valid",
            use_bias=True,
            dilation_rate=1,
            activation=None,
        )
        input_blob = flow.math.sigmoid(input=input_blob)
        bn3 = flow.math.multiply(x=input_blob, y=bn3)
        # se end

    if dim_match:
        input_blob = in_data
    else:
        input_blob = _conv2d_layer(
            name="%s%s_conv1sc" % (name, suffix),
            input=in_data,
            filters=num_filter,
            kernel_size=1,
            strides=stride,
            padding="valid",
            use_bias=False,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=bn_is_training,
            name="%s%s_sc" % (name, suffix),
        )

    identity = flow.math.add(x=bn3, y=input_blob)
    return identity


def Resnet100(input_blob, embedding_size, fc_type="GDC", bn_is_training=True, **kw):
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    units = [3, 13, 30, 3]

    input_blob = _conv2d_layer(
        name="conv0",
        input=input_blob,
        filters=filter_list[0],
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    input_blob = _batch_norm(
        input_blob, epsilon=2e-5, is_training=bn_is_training, name="bn0"
    )
    input_blob = _prelu(input_blob, name="relu0")

    for i in range(num_stages):
        input_blob = residual_unit_v3(
            input_blob,
            filter_list[i + 1],
            [2, 2],
            False,
            bn_is_training=bn_is_training,
            name="stage%d_unit%d" % (i + 1, 1),
        )
        for j in range(units[i] - 1):
            input_blob = residual_unit_v3(
                input_blob,
                filter_list[i + 1],
                [1, 1],
                True,
                bn_is_training=bn_is_training,
                name="stage%d_unit%d" % (i + 1, j + 2),
            )
    if fc_type == "GDC":
        input_blob = Linear(
            input_blob,
            num_filter=512,
            num_group=512,
            kernel=7,
            pad="valid",
            stride=[1, 1],
            bn_is_training=bn_is_training,
            name="conv_6dw7_7",
        )
        input_blob = flow.reshape(input_blob, (input_blob.shape[0], -1))
        pre_fc1 = flow.layers.dense(
            inputs=input_blob,
            units=embedding_size,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer(),
            bias_regularizer=_get_regularizer(),
            trainable=True,
            name="pre_fc1",
        )
        fc1 = _batch_norm(
            pre_fc1,
            epsilon=2e-5,
            center=True,
            scale=False,
            is_training=bn_is_training,
            name="fc1",
        )

    elif fc_type == "E":
        input_blob = _batch_norm(
            input_blob, epsilon=2e-5, is_training=bn_is_training, name="bn1"
        )
        input_blob = _dropout(input_blob, dropout_prob=0.4)
        input_blob = flow.reshape(input_blob, (input_blob.shape[0], -1))
        pre_fc1 = flow.layers.dense(
            inputs=input_blob,
            units=embedding_size,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer(),
            bias_regularizer=_get_regularizer(),
            trainable=True,
            name="pre_fc1",
        )
        fc1 = _batch_norm(
            pre_fc1,
            epsilon=2e-5,
            center=True,
            scale=False,
            is_training=bn_is_training,
            name="fc1",
        )
    elif fc_type == "FC":
        input_blob = _batch_norm(
            input_blob, epsilon=2e-5, is_training=bn_is_training, name="bn1"
        )
        input_blob = flow.reshape(input_blob, (input_blob.shape[0], -1))
        pre_fc1 = flow.layers.dense(
            inputs=input_blob,
            units=embedding_size,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=_get_regularizer(),
            bias_regularizer=_get_regularizer(),
            trainable=True,
            name="pre_fc1",
        )
        fc1 = _batch_norm(
            pre_fc1,
            epsilon=2e-5,
            center=True,
            scale=False,
            is_training=bn_is_training,
            name="fc1",
        )

    else:
        print("unimplemented")
    return fc1
