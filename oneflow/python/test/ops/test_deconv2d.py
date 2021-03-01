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
import test_global_storage
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, params_case, dilations, data_format):
    input_shape, output_shape, padding, strides, kernel_size = params_case
    assert data_format in ["NCHW", "NHWC"]
    out_channels = output_shape[1] if data_format == "NCHW" else output_shape[3]
    in_channels = input_shape[1] if data_format == "NCHW" else input_shape[3]
    assert device_type in ["gpu"]

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def DeconvJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            if data_format == "NCHW":
                weight = flow.get_variable(
                    "weight",
                    shape=(in_channels, out_channels, kernel_size, kernel_size),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                    trainable=True,
                )
            else:
                weight = flow.get_variable(
                    "weight",
                    shape=(in_channels, kernel_size, kernel_size, out_channels),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                    trainable=True,
                )
            loss = flow.nn.conv2d_transpose(
                x,
                weight,
                strides=strides,
                output_shape=output_shape,
                dilations=dilations,
                padding=padding,
                data_format=data_format,
            )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(weight, test_global_storage.Setter("weight"))
            flow.watch_diff(weight, test_global_storage.Setter("weight_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    of_out = DeconvJob().get()
    # Tensorflow
    if data_format == "NCHW":
        with tf.GradientTape(persistent=True) as tape:
            x = tf.Variable(test_global_storage.Get("x").transpose(0, 2, 3, 1))
            output_shape = (
                output_shape[0],
                output_shape[2],
                output_shape[3],
                output_shape[1],
            )
            w = tf.Variable(test_global_storage.Get("weight").transpose(2, 3, 1, 0))
            tf_out = tf.nn.conv2d_transpose(
                x,
                w,
                output_shape=output_shape,
                strides=[1, strides, strides, 1],
                padding=padding,
                data_format="NHWC",
            )

        loss_diff = test_global_storage.Get("loss_diff").transpose(0, 2, 3, 1)
        tf_x_diff = tape.gradient(tf_out, x, loss_diff)
        tf_weight_diff = tape.gradient(tf_out, w, loss_diff)

        assert np.allclose(
            of_out.numpy().transpose(0, 2, 3, 1), tf_out.numpy(), rtol=1e-02, atol=1e-02
        )
        assert np.allclose(
            test_global_storage.Get("x_diff").transpose(0, 2, 3, 1),
            tf_x_diff.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        assert np.allclose(
            test_global_storage.Get("weight_diff").transpose(2, 3, 1, 0),
            tf_weight_diff.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
    else:
        with tf.GradientTape(persistent=True) as tape:
            x = tf.Variable(test_global_storage.Get("x"))
            w = tf.Variable(test_global_storage.Get("weight").transpose(1, 2, 3, 0))
            tf_out = tf.nn.conv2d_transpose(
                x,
                w,
                output_shape=output_shape,
                strides=[1, strides, strides, 1],
                padding=padding,
                data_format="NHWC",
            )
        loss_diff = test_global_storage.Get("loss_diff")
        tf_x_diff = tape.gradient(tf_out, x, loss_diff)
        tf_weight_diff = tape.gradient(tf_out, w, loss_diff)

        assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-02, atol=1e-02), (
            of_out.numpy() - tf_out.numpy()
        )
        assert np.allclose(
            test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-02, atol=1e-02
        )
        assert np.allclose(
            test_global_storage.Get("weight_diff").transpose(1, 2, 3, 0),
            tf_weight_diff.numpy(),
            rtol=1e-2,
            atol=1e-2,
        )


@flow.unittest.skip_unless_1n1d()
class TestDeconv2d(flow.unittest.TestCase):
    def test_deconv2d_NHWC_1n1c(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        # params_case: (input_shape, output_shape, padding, stirdes, kernel_size)
        arg_dict["params_case"] = [
            ((32, 3, 3, 4), (32, 3, 3, 8), "SAME", 1, 3),
            ((32, 3, 3, 2), (32, 6, 6, 8), "SAME", 2, 4),
            ((32, 2, 2, 1), (32, 5, 5, 2), "VALID", 2, 2),
            ((32, 2, 2, 16), (32, 8, 8, 4), "VALID", 2, 5),
        ]
        arg_dict["dilations"] = [1]
        arg_dict["data_format"] = ["NHWC"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_deconv2d_NCHW_1n1c(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        # params_case: (input_shape, output_shape, padding, stirdes, kernel_size)
        arg_dict["params_case"] = [
            ((32, 4, 3, 3), (32, 8, 3, 3), "SAME", 1, 3),
            ((32, 4, 3, 3), (32, 8, 6, 6), "SAME", 2, 5),
            ((32, 1, 2, 2), (32, 2, 5, 5), "VALID", 2, 2),
            ((32, 16, 2, 2), (32, 4, 8, 8), "VALID", 2, 5),
        ]
        arg_dict["dilations"] = [1]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
