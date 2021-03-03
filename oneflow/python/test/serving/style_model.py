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


def instance_norm(input, name_prefix, trainable=True):
    (mean, variance) = flow.nn.moments(input, [2, 3], keepdims=True)
    gamma = flow.get_variable(
        name_prefix + "_gamma",
        shape=(1, input.shape[1], 1, 1),
        dtype=input.dtype,
        initializer=flow.ones_initializer(),
        trainable=trainable,
    )
    beta = flow.get_variable(
        name_prefix + "_beta",
        shape=(1, input.shape[1], 1, 1),
        dtype=input.dtype,
        initializer=flow.zeros_initializer(),
        trainable=trainable,
    )
    epsilon = 1e-3
    normalized = (input - mean) / flow.math.sqrt(variance + epsilon)
    return gamma * normalized + beta


def conv2d_layer(
    name,
    input,
    out_channel,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    use_bias=True,
    weight_initializer=flow.variance_scaling_initializer(
        2, "fan_out", "random_normal", data_format="NCHW"
    ),
    bias_initializer=flow.zeros_initializer(),
    trainable=True,
):
    weight_shape = (out_channel, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "_weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        trainable=trainable,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "_bias",
            shape=(out_channel,),
            dtype=input.dtype,
            initializer=bias_initializer,
            trainable=trainable,
        )
        output = flow.nn.bias_add(output, bias, data_format)
    return output


def upsampleConvLayer(
    input,
    name_prefix,
    channel,
    kernel_size,
    hw_scale=(2, 2),
    data_format="NCHW",
    # interpolation = "bilinear",
    interpolation="nearest",
    trainable=True,
):
    upsample = flow.layers.upsample_2d(
        input,
        size=hw_scale,
        data_format=data_format,
        interpolation=interpolation,
        name=name_prefix + "_%s" % interpolation,
    )
    return conv2d_layer(
        name_prefix + "_conv",
        upsample,
        channel,
        kernel_size=kernel_size,
        strides=1,
        trainable=trainable,
    )


def resBlock(input, channel, name_prefix, trainable=True):
    out = conv2d_layer(
        name_prefix + "_conv1",
        input,
        channel,
        kernel_size=3,
        strides=1,
        trainable=trainable,
    )
    out = instance_norm(out, name_prefix + "_in1", trainable=trainable)
    out = flow.nn.relu(out)
    out = conv2d_layer(
        name_prefix + "_conv2",
        out,
        channel,
        kernel_size=3,
        strides=1,
        trainable=trainable,
    )
    out = instance_norm(out, name_prefix + "_in2", trainable=trainable)
    return out + input


def deconv(
    input, out_channel, name_prefix, kernel_size=4, strides=[2, 2], trainable=True
):
    weight = flow.get_variable(
        name_prefix + "_weight",
        shape=(input.shape[1], out_channel, kernel_size, kernel_size),
        dtype=flow.float,
        initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
        trainable=True,
    )
    return flow.nn.conv2d_transpose(
        input,
        weight,
        strides=strides,
        padding="SAME",
        output_shape=(
            input.shape[0],
            out_channel,
            input.shape[2] * strides[0],
            input.shape[3] * strides[1],
        ),
    )


def styleNet(input, trainable=True):
    with flow.scope.namespace("style_transfer"):
        # Initial convolution layers
        conv1 = conv2d_layer(
            "first_conv", input, 32, kernel_size=9, strides=1, trainable=trainable
        )
        in1 = instance_norm(conv1, "first_conv_in", trainable=trainable)
        in1 = flow.nn.relu(in1)
        conv2 = conv2d_layer(
            "second_conv", in1, 64, kernel_size=3, strides=2, trainable=trainable
        )
        in2 = instance_norm(conv2, "second_conv_in", trainable=trainable)
        in2 = flow.nn.relu(in2)
        conv3 = conv2d_layer(
            "third_conv", in2, 128, kernel_size=3, strides=2, trainable=trainable
        )
        in3 = instance_norm(conv3, "third_conv_in", trainable=trainable)
        in3 = flow.nn.relu(in3)
        # Residual layers
        res1 = resBlock(in3, 128, "res1", trainable=trainable)
        res2 = resBlock(res1, 128, "res2", trainable=trainable)
        res3 = resBlock(res2, 128, "res3", trainable=trainable)
        res4 = resBlock(res3, 128, "res4", trainable=trainable)
        res5 = resBlock(res4, 128, "res5", trainable=trainable)
        # Upsampling Layers
        upsample1 = upsampleConvLayer(res5, "upsample1", 64, 3, trainable=trainable)
        # upsample1 = deconv(res5, 64, "upsample1", kernel_size = 4, strides = [2, 2], trainable = True)
        in4 = instance_norm(upsample1, "upsample1_in", trainable=trainable)
        in4 = flow.nn.relu(in4)
        upsample2 = upsampleConvLayer(in4, "upsample2", 32, 3, trainable=trainable)
        # upsample2 = deconv(in4, 32, "upsample2", kernel_size = 4, strides = [2, 2], trainable = True)
        in5 = instance_norm(upsample2, "upsample2_in", trainable=trainable)
        in5 = flow.nn.relu(in5)
        conv1 = conv2d_layer(
            "last_conv", in5, 3, kernel_size=9, strides=1, trainable=trainable
        )
        out = flow.clamp(conv1, 0, 255)
    return out


def gram_matrix(input):
    b = input.shape[0]
    ch = input.shape[1]
    h = input.shape[2]
    w = input.shape[3]
    features = flow.reshape(input, [b, ch, h * w])
    features_t = flow.transpose(features, [0, 2, 1])
    gram = flow.matmul(features, features_t) / (ch * h * w)
    return gram


def mse_loss(input):
    return flow.math.reduce_mean(flow.math.square(input))
