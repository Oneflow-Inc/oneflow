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

import os
import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import test_global_storage
from test_util import GenArgList

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def grouped_convolution2D(
    inputs, filters, padding, num_groups, strides=None, dilation_rate=None
):
    input_list = tf.split(inputs, num_groups, axis=-1)
    filter_list = tf.split(filters, num_groups, axis=-1)
    output_list = []
    for (conv_idx, (input_tensor, filter_tensor)) in enumerate(
        zip(input_list, filter_list)
    ):
        output_list.append(
            tf.nn.conv2d(
                input_tensor,
                filter_tensor,
                padding="VALID",
                strides=[1, 1, 1, 1],
                data_format="NHWC",
            )
        )
    outputs = tf.concat(output_list, axis=-1)
    return outputs


def compare_with_tensorflow(
    device_type,
    x_shape,
    filters,
    kernel_size,
    groups,
    data_format="NCHW",
    padding="VALID",
    stride=1,
):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    flow.clear_default_session()
    if data_format == "NCHW":
        xy_data_transpose = (0, 2, 3, 1)
        weight_data_transpose = (2, 3, 1, 0)
    else:
        xy_data_transpose = (0, 1, 2, 3)
        weight_data_transpose = (1, 2, 3, 0)

    @flow.global_function(type="train", function_config=func_config)
    def RunConvBias():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            if data_format == "NCHW":
                weight_shape = (filters, x.shape[1] // groups, kernel_size, kernel_size)
            else:
                weight_shape = (filters, kernel_size, kernel_size, x.shape[3] // groups)
            weight = flow.get_variable(
                "conv-weight",
                shape=weight_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
            )
            bias = flow.get_variable(
                "conv-bias",
                shape=(filters,),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
            )
            loss = flow.nn.conv2d(
                x,
                weight,
                bias=bias,
                strides=[stride, stride],
                padding=padding,
                dilations=[1, 1],
                groups=groups,
                name="conv",
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(weight, test_global_storage.Setter("weight"))
            flow.watch_diff(weight, test_global_storage.Setter("weight_diff"))
            flow.watch(bias, test_global_storage.Setter("bias"))
            flow.watch_diff(bias, test_global_storage.Setter("bias_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    of_out = RunConvBias().get()
    flow.clear_default_session()
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x").transpose(xy_data_transpose))
        assert groups > 0
        assert filters % groups == 0
        if groups == 1:
            weight = tf.Variable(
                test_global_storage.Get("weight").transpose(weight_data_transpose)
            )
            conv_out = tf.nn.conv2d(
                x,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding,
                data_format="NHWC",
            )
        else:
            weight = tf.Variable(
                test_global_storage.Get("weight").transpose(weight_data_transpose)
            )
            conv_out = grouped_convolution2D(
                x, weight, padding=padding, num_groups=groups
            )
        bias = tf.Variable(test_global_storage.Get("bias"))
        tf_out = tf.nn.bias_add(conv_out, bias, data_format="NHWC")
    loss_diff = test_global_storage.Get("loss_diff").transpose(xy_data_transpose)
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    tf_weight_diff = tape.gradient(tf_out, weight, loss_diff)
    tf_bias_diff = tape.gradient(tf_out, bias, loss_diff)
    max_diff = np.max(
        np.absolute(of_out.numpy().transpose(xy_data_transpose) - tf_out.numpy())
    )
    assert np.allclose(
        of_out.numpy().transpose(xy_data_transpose),
        tf_out.numpy(),
        rtol=0.005,
        atol=0.005,
    ), max_diff
    assert np.allclose(
        test_global_storage.Get("x_diff").transpose(xy_data_transpose),
        tf_x_diff.numpy(),
        rtol=0.005,
        atol=0.005,
    )
    assert np.allclose(
        test_global_storage.Get("weight_diff").transpose(weight_data_transpose),
        tf_weight_diff.numpy(),
        rtol=0.005,
        atol=0.005,
    )
    assert np.allclose(
        test_global_storage.Get("bias_diff"),
        tf_bias_diff.numpy(),
        rtol=0.005,
        atol=0.005,
    )


@flow.unittest.skip_unless_1n1d()
class TestNnConv2dBias(flow.unittest.TestCase):
    def test_cpu_1x1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["x_shape"] = [(3, 32, 128, 128)]
        arg_dict["filters"] = [5]
        arg_dict["kernel_size"] = [1]
        arg_dict["groups"] = [1]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_cpu_depthwise(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["x_shape"] = [(10, 32, 10, 10)]
        arg_dict["filters"] = [128]
        arg_dict["kernel_size"] = [1]
        arg_dict["groups"] = [32]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_cpu_group(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["x_shape"] = [(10, 32, 226, 226)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [1]
        arg_dict["groups"] = [4]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_cpu_5x5(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [5]
        arg_dict["groups"] = [1]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_gpu_1x1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [1]
        arg_dict["groups"] = [8]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_gpu_depthwise(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3]
        arg_dict["groups"] = [32]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_gpu_group(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3]
        arg_dict["groups"] = [4]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_gpu_5x5(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 32, 20, 20)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [5]
        arg_dict["groups"] = [1]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
