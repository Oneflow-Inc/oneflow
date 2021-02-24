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
from test_util import GenArgList, type_name_to_flow_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, x_shape, data_type, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def SoftmaxJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=True,
            )
            x1 = x
            x = flow.identity(x)
            if data_type == "float16":
                loss = flow.cast(
                    flow.nn.softmax(flow.cast(x, dtype=flow.float16), axis=axis),
                    dtype=flow.float,
                )
            else:
                loss = flow.nn.softmax(x, axis=axis)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            total_loss = loss * x1

            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(total_loss)

            return loss

    # OneFlow
    of_out = SoftmaxJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.nn.softmax(x, axis=axis)

    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    if data_type == "float16":
        tolerance = 1e-3
    else:
        tolerance = 1e-5
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=tolerance, atol=tolerance)
    assert np.allclose(
        test_global_storage.Get("x_diff"),
        tf_x_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


@flow.unittest.skip_unless_1n1d()
class TestSoftmax(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_softmax_shape(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [
            (10, 10, 20, 30),
            (10, 20, 13),
            (10, 20, 30),
            (10, 20),
            (10, 60),
            (32, 12, 128),
            (10, 960),
            (12, 2001),
            (10, 4096),
            (10, 8092),
            (256, 1001),
            (100, 65536),
            (10, 65535),
        ]
        arg_dict["data_type"] = ["float32", "double", "float16"]
        arg_dict["axis"] = [-1]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            compare_with_tensorflow(*arg)

    def test_softmax_axis(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [(10, 20, 30, 40)]
        arg_dict["data_type"] = ["float32", "double", "float16"]
        arg_dict["axis"] = [-4, -3, -2, -1, 0, 1, 2, 3]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
