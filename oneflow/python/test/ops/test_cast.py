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
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def cast_forward_compare_with_tensorflow(test_cast, device_type, input_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def cast_forward(
        input_def: oft.Numpy.Placeholder(
            shape=input_shape, dtype=type_name_to_flow_type[dtype]
        )
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.cast(input_def, dtype=type_name_to_flow_type[dtype])

    input = np.random.rand(*input_shape).astype(type_name_to_np_type[dtype])
    of_out = cast_forward(input).get()
    tf_out = tf.cast(input, dtype=type_name_to_np_type[dtype])
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-5, atol=1e-5)


def compare_with_tensorflow(device_type, input_shape, dtype):
    assert device_type in ["gpu", "cpu"]
    assert dtype in ["float32", "double"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def CastJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=type_name_to_flow_type[dtype],
                initializer=flow.random_uniform_initializer(),
                trainable=True,
            )

            loss = flow.cast(x, dtype=flow.float)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = CastJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.cast(x, dtype=type_name_to_np_type[dtype])
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n1d()
class TestCast(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(5, 4, 3)]
        arg_dict["dtype"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_cast_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(5, 4, 3)]
        arg_dict["dtype"] = ["float32", "int8", "uint8", "double", "int32", "int64"]
        for arg in GenArgList(arg_dict):
            cast_forward_compare_with_tensorflow(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
