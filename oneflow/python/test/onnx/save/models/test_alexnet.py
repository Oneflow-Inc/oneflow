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
import oneflow as flow
import oneflow.typing as tp
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import onnxruntime as ort
import onnx
from oneflow.python.test.onnx.save.util import convert_to_onnx_and_check


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):
    weight_shape = (filters, input.shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output


def alexnet(images, labels, trainable=True):
    transposed = flow.transpose(images, name="transpose", perm=[0, 3, 1, 2])
    conv1 = _conv2d_layer(
        "conv1", transposed, filters=64, kernel_size=11, strides=4, padding="VALID"
    )

    pool1 = flow.nn.avg_pool2d(conv1, 3, 2, "VALID", "NCHW", name="pool1")

    conv2 = _conv2d_layer("conv2", pool1, filters=192, kernel_size=5)

    pool2 = flow.nn.avg_pool2d(conv2, 3, 2, "VALID", "NCHW", name="pool2")

    conv3 = _conv2d_layer("conv3", pool2, filters=384)

    conv4 = _conv2d_layer("conv4", conv3, filters=384)

    conv5 = _conv2d_layer("conv5", conv4, filters=256)

    pool5 = flow.nn.avg_pool2d(conv5, 3, 2, "VALID", "NCHW", name="pool5")

    def _get_initializer():
        return flow.random_uniform_initializer()

    if len(pool5.shape) > 2:
        pool5 = flow.reshape(pool5, shape=(pool5.shape[0], -1))

    fc1 = flow.layers.dense(
        inputs=pool5,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc1",
    )

    dropout1 = fc1

    fc2 = flow.layers.dense(
        inputs=dropout1,
        units=4096,
        activation=flow.math.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc2",
    )

    dropout2 = fc2

    fc3 = flow.layers.dense(
        inputs=dropout2,
        units=1001,
        activation=None,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=trainable,
        name="fc3",
    )

    return fc3


def test_alexnet(test_case):
    @flow.global_function()
    def alexnet_eval_job(x: tp.Numpy.Placeholder((1, 227, 227, 3))):
        return alexnet(x, None, False)

    convert_to_onnx_and_check(alexnet_eval_job)
