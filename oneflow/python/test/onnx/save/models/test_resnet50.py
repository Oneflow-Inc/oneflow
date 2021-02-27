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

import oneflow as flow
import oneflow.typing as tp
import onnx
import onnxruntime as ort
import numpy as np
from oneflow.python.test.onnx.save.util import convert_to_onnx_and_check


BLOCK_COUNTS = [3, 4, 6, 3]
BLOCK_FILTERS = [256, 512, 1024, 2048]
BLOCK_FILTERS_INNER = [64, 128, 256, 512]

g_trainable = False


def _conv2d(
    name,
    input,
    filters,
    kernel_size,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilations=1,
    trainable=True,
    # weight_initializer=flow.variance_scaling_initializer(data_format="NCHW"),
    weight_initializer=flow.variance_scaling_initializer(
        2, "fan_in", "random_normal", data_format="NCHW"
    ),
    weight_regularizer=flow.regularizers.l2(1.0 / 32768),
):
    weight = flow.get_variable(
        name + "-weight",
        shape=(filters, input.shape[1], kernel_size, kernel_size),
        dtype=input.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        model_name="weight",
        trainable=trainable,
    )
    return flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilations, name=name
    )


def _batch_norm(inputs, name=None, trainable=True):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        momentum=0.9,  # 97,
        epsilon=1.001e-5,
        center=True,
        scale=True,
        trainable=trainable,
        training=trainable,
        name=name,
    )


def conv2d_affine(input, name, filters, kernel_size, strides, activation=None):
    # input data_format must be NCHW, cannot check now
    padding = "SAME" if strides > 1 or kernel_size > 1 else "VALID"
    output = _conv2d(
        name, input, filters, kernel_size, strides, padding, trainable=g_trainable
    )
    output = _batch_norm(output, name + "_bn", trainable=g_trainable)
    if activation == "Relu":
        output = flow.math.relu(output)

    return output


def bottleneck_transformation(input, block_name, filters, filters_inner, strides):
    a = conv2d_affine(
        input, block_name + "_branch2a", filters_inner, 1, 1, activation="Relu",
    )

    b = conv2d_affine(
        a, block_name + "_branch2b", filters_inner, 3, strides, activation="Relu",
    )

    c = conv2d_affine(b, block_name + "_branch2c", filters, 1, 1)

    return c


def residual_block(input, block_name, filters, filters_inner, strides_init):
    if strides_init != 1 or block_name == "res2_0":
        shortcut = conv2d_affine(
            input, block_name + "_branch1", filters, 1, strides_init
        )
    else:
        shortcut = input

    bottleneck = bottleneck_transformation(
        input, block_name, filters, filters_inner, strides_init
    )

    return flow.math.relu(bottleneck + shortcut)


def residual_stage(input, stage_name, counts, filters, filters_inner, stride_init=2):
    output = input
    for i in range(counts):
        block_name = "%s_%d" % (stage_name, i)
        output = residual_block(
            output, block_name, filters, filters_inner, stride_init if i == 0 else 1,
        )

    return output


def resnet_conv_x_body(input, on_stage_end=lambda x: x):
    output = input
    for i, (counts, filters, filters_inner) in enumerate(
        zip(BLOCK_COUNTS, BLOCK_FILTERS, BLOCK_FILTERS_INNER)
    ):
        stage_name = "res%d" % (i + 2)
        output = residual_stage(
            output, stage_name, counts, filters, filters_inner, 1 if i == 0 else 2,
        )
        on_stage_end(output)

    return output


def resnet_stem(input):
    conv1 = _conv2d("conv1", input, 1, 1, 2)
    tmp = _batch_norm(conv1, "conv1_bn", trainable=g_trainable)
    conv1_bn = flow.math.relu(tmp)
    pool1 = flow.nn.max_pool2d(
        conv1_bn, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="pool1",
    )
    return pool1


def resnet50(images, trainable=True, need_transpose=False):

    # note: images.shape = (N C H W) in cc's new dataloader, transpose is not needed anymore
    if need_transpose:
        images = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])

    with flow.scope.namespace("Resnet"):
        stem = resnet_stem(images)
        body = resnet_conv_x_body(stem, lambda x: x)
        pool5 = flow.nn.avg_pool2d(
            body, ksize=7, strides=1, padding="VALID", data_format="NCHW", name="pool5",
        )

        fc1001 = flow.layers.dense(
            flow.reshape(pool5, (pool5.shape[0], -1)),
            units=1000,
            use_bias=True,
            kernel_initializer=flow.variance_scaling_initializer(
                2, "fan_in", "random_normal"
            ),
            # kernel_initializer=flow.xavier_uniform_initializer(),
            bias_initializer=flow.random_uniform_initializer(),
            kernel_regularizer=flow.regularizers.l2(1.0 / 32768),
            trainable=trainable,
            name="fc1001",
        )

    return fc1001


def test_resnet50(test_case):
    @flow.global_function()
    def InferenceNet(images: tp.Numpy.Placeholder((1, 3, 224, 224))):
        logits = resnet50(images)

        predictions = flow.nn.softmax(logits)
        return predictions

    convert_to_onnx_and_check(InferenceNet)
