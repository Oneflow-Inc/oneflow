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
import oneflow_api
import tensorflow as tf
import test_global_storage
from test_util import GenArgList
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_reduce_sum_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def ReduceSumJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )
            loss = flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    # OneFlow
    of_out = ReduceSumJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n2d()
class TestReduceOpsV2(flow.unittest.TestCase):
    def test_reduce_sum_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(*arg)

    def test_reduce_sum_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(*arg)

    def test_reduce_sum_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(*arg)

    def test_reduce_sum_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(*arg)

    def test_reduce_sum_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_sum(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
