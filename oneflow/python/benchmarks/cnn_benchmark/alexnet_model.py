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
from model_util import conv2d_layer


def alexnet(images, trainable=True):

    conv1 = conv2d_layer(
        "conv1", images, filters=64, kernel_size=11, strides=4, padding="VALID"
    )

    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = conv2d_layer("conv3", pool2, filters=384)

    conv4 = conv2d_layer("conv4", conv3, filters=384)

    conv5 = conv2d_layer("conv5", conv4, filters=256)

    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc1",
    )

    dropout1 = flow.nn.dropout(fc1, rate=0.5)

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc2",
    )

    dropout2 = flow.nn.dropout(fc2, rate=0.5)

    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=flow.random_uniform_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc3",
    )

    return fc3
