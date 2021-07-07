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
from __future__ import absolute_import, division, print_function

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
from model_util import conv2d_layer


def _conv_block(in_blob, index, filters, conv_times):
    conv_block = []
    conv_block.insert(0, in_blob)
    for i in range(conv_times):
        conv_i = conv2d_layer(
            name="conv{}".format(index),
            input=conv_block[i],
            filters=filters,
            kernel_size=3,
            strides=1,
        )
        conv_block.append(conv_i)
        index += 1

    return conv_block


def vgg16(images, trainable=True):
    conv1 = _conv_block(images, 0, 64, 2)
    pool1 = flow.nn.max_pool2d(conv1[-1], 2, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv_block(pool1, 2, 128, 2)
    pool2 = flow.nn.max_pool2d(conv2[-1], 2, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv_block(pool2, 4, 256, 3)
    pool3 = flow.nn.max_pool2d(conv3[-1], 2, 2, "VALID", "NCHW", name="pool3")

    conv4 = _conv_block(pool3, 7, 512, 3)
    pool4 = flow.nn.max_pool2d(conv4[-1], 2, 2, "VALID", "NCHW", name="pool4")

    conv5 = _conv_block(pool4, 10, 512, 3)
    pool5 = flow.nn.max_pool2d(conv5[-1], 2, 2, "VALID", "NCHW", name="pool5")

    def _get_kernel_initializer():
        kernel_initializer = initializer_conf_util.InitializerConf()
        kernel_initializer.truncated_normal_conf.std = 0.816496580927726
        return kernel_initializer

    def _get_bias_initializer():
        bias_initializer = initializer_conf_util.InitializerConf()
        bias_initializer.constant_conf.value = 0.0
        return bias_initializer

    pool5 = flow.reshape(pool5, [pool5.shape[0], -1])

    fc6 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.math.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc1",
    )

    fc6 = flow.nn.dropout(fc6, rate=0.5)

    fc7 = flow.layers.dense(
        inputs=fc6,
        units=4096,
        activation=flow.math.relu,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc2",
    )
    fc7 = flow.nn.dropout(fc7, rate=0.5)

    fc8 = flow.layers.dense(
        inputs=fc7,
        units=1001,
        use_bias=True,
        kernel_initializer=_get_kernel_initializer(),
        bias_initializer=_get_bias_initializer(),
        trainable=trainable,
        name="fc_final",
    )

    return fc8
