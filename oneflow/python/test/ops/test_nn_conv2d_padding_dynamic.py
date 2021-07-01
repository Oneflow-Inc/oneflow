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
import unittest
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

global_storage = {}


def global_storage_setter(name):
    global global_storage

    def _set(x):
        global_storage[name] = x

    return _set


def compare_with_tensorflow(
    device_type,
    x_shape,
    filters,
    kernel_size,
    groups,
    of_padding="SAME",
    tf_padding="SAME",
    stride=1,
    data_format="NCHW",
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    if data_format == "NCHW":
        xy_data_transpose = (0, 2, 3, 1)
        weight_data_transpose = (2, 3, 1, 0)
    else:
        xy_data_transpose = (0, 1, 2, 3)
        weight_data_transpose = (1, 2, 3, 0)

    @flow.global_function(type="train", function_config=func_config)
    def DynamicConvJob(x: oft.ListNumpy.Placeholder((10, 3, 100, 100))):
        with flow.scope.placement(device_type, "0:0"):
            x_var = flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            x_var = flow.cast_to_current_logical_view(x_var)
            x += x_var
            if data_format == "NCHW":
                weight_shape = (filters, x_shape[1] // groups, kernel_size, kernel_size)
            else:
                weight_shape = (filters, kernel_size, kernel_size, x_shape[3] // groups)
            weight = flow.get_variable(
                "conv-weight",
                shape=weight_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
            )
            weight = flow.cast_to_current_logical_view(weight)
            loss = flow.nn.conv2d(
                x,
                weight,
                strides=[stride, stride],
                padding=of_padding,
                data_format=data_format,
                dilations=[1, 1],
                groups=groups,
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, global_storage_setter("x"))
            flow.watch_diff(x, global_storage_setter("x_diff"))
            flow.watch(weight, global_storage_setter("weight"))
            flow.watch_diff(weight, global_storage_setter("weight_diff"))
            flow.watch(loss, global_storage_setter("loss"))
            flow.watch_diff(loss, global_storage_setter("loss_diff"))

            return loss

    # OneFlow
    data = [np.random.rand(*x_shape).astype(np.float32)]
    of_out = DynamicConvJob(data).get().numpy_list()[0]
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(data[0].transpose(xy_data_transpose))
        assert groups > 0
        assert x_shape[1] % groups == 0
        assert filters % groups == 0
        weight = tf.Variable(
            global_storage["weight"].numpy().transpose(weight_data_transpose)
        )

        tf_out = tf.nn.conv2d(
            x,
            weight,
            strides=[1, stride, stride, 1],
            padding=tf_padding,
            data_format="NHWC",
        )

    idx = np.where(np.abs(of_out.transpose(xy_data_transpose) - tf_out.numpy()) > 5e-4)
    assert np.allclose(
        of_out.transpose(xy_data_transpose), tf_out.numpy(), rtol=1e-3, atol=1e-3,
    )

    loss_diff = global_storage["loss_diff"].numpy_list()[0].transpose(xy_data_transpose)
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    tf_weight_diff = tape.gradient(tf_out, weight, loss_diff)
    rtol = 1e-4
    atol = 1e-4
    if device_type == "cpu":
        rtol *= 100
        atol *= 100
    assert np.allclose(
        global_storage["x_diff"].numpy_list()[0].transpose(xy_data_transpose),
        tf_x_diff.numpy(),
        rtol=rtol,
        atol=atol,
    ), (
        global_storage["x_diff"].numpy_list()[0].transpose(xy_data_transpose)
        - tf_x_diff.numpy()
    )
    assert np.allclose(
        global_storage["weight_diff"].numpy().transpose(weight_data_transpose),
        tf_weight_diff.numpy(),
        rtol=5e-3,
        atol=5e-3,
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skip("skip_for_ci")
class TestNnConv2dPaddingDynamic(flow.unittest.TestCase):
    def test_padding_valid(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = ["VALID"]
        arg_dict["tf_padding"] = ["VALID"]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_padding_same(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = ["SAME_UPPER"]
        arg_dict["tf_padding"] = ["SAME"]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_pad_list1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = [[[0, 0], [0, 0], [0, 1], [1, 0]]]
        arg_dict["tf_padding"] = [[[0, 0], [0, 1], [1, 0], [0, 0]]]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_pad_list2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = [[[0, 0], [0, 0], [1, 1], [1, 1]]]
        arg_dict["tf_padding"] = [[[0, 0], [1, 1], [1, 1], [0, 0]]]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_pad_list3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = [[[0, 0], [0, 0], [1, 0], [1, 0]]]
        arg_dict["tf_padding"] = [[[0, 0], [1, 0], [1, 0], [0, 0]]]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_pad_list4(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(10, 3, 10, 10), (10, 3, 11, 11)]
        arg_dict["filters"] = [64]
        arg_dict["kernel_size"] = [3, 2]
        arg_dict["groups"] = [1]
        arg_dict["of_padding"] = [[[0, 0], [0, 0], [10, 2], [10, 2]]]
        arg_dict["tf_padding"] = [[[0, 0], [10, 2], [10, 2], [0, 0]]]
        arg_dict["stride"] = [1, 2]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
