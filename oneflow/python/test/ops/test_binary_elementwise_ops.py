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
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict
import oneflow.typing as oft

import test_global_storage
from test_util import (
    GenArgDict,
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowBinaryOp(device_type, flow_op, x, y, data_type):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    flow_type = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(
        x: oft.Numpy.Placeholder(x.shape, dtype=flow_type),
        y: oft.Numpy.Placeholder(y.shape, dtype=flow_type),
    ):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="x",
                shape=x.shape,
                dtype=flow_type,
                initializer=flow.zeros_initializer(),
                trainable=True,
            )
            y += flow.get_variable(
                name="y",
                shape=y.shape,
                dtype=flow_type,
                initializer=flow.zeros_initializer(),
                trainable=True,
            )
            loss = flow_op(x, y)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch_diff(y, test_global_storage.Setter("y_diff"))

            return loss

    # Oneflow
    out = FlowJob(x, y).get().numpy()
    x_diff = test_global_storage.Get("x_diff")
    y_diff = test_global_storage.Get("y_diff")
    return out, x_diff, y_diff


def RunTensorFlowBinaryOp(tf_op, x, y):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = tf.Variable(y)
        out = tf_op(x, y)
    x_diff = tape.gradient(out, x)
    y_diff = tape.gradient(out, y)
    return out.numpy(), x_diff, y_diff


def compare_with_tensorflow(
    test_case,
    device_type,
    flow_op,
    tf_op,
    x_shape,
    y_shape,
    data_type,
    x_minval=-10,
    x_maxval=10,
    y_minval=-10,
    y_maxval=10,
    compare_grad=True,
    out_rtol=1e-5,
    out_atol=1e-5,
    diff_rtol=1e-5,
    diff_atol=1e-5,
):
    test_case.assertTrue(device_type in ["gpu", "cpu"])

    np_type = type_name_to_np_type[data_type]
    x = np.random.uniform(low=x_minval, high=x_maxval, size=x_shape).astype(np_type)
    y = np.random.uniform(low=y_minval, high=y_maxval, size=y_shape).astype(np_type)

    of_out, of_x_diff, of_y_diff, = RunOneflowBinaryOp(
        device_type, flow_op, x, y, data_type
    )
    tf_out, tf_x_diff, tf_y_diff = RunTensorFlowBinaryOp(tf_op, x, y)

    test_case.assertTrue(
        np.allclose(of_out, tf_out, rtol=out_rtol, atol=out_atol, equal_nan=True)
    )
    if compare_grad:
        test_case.assertTrue(
            np.allclose(
                of_x_diff,
                tf_x_diff.numpy(),
                rtol=diff_rtol,
                atol=diff_atol,
                equal_nan=True,
            )
        )
        test_case.assertTrue(
            np.allclose(
                of_y_diff,
                tf_y_diff.numpy(),
                rtol=diff_rtol,
                atol=diff_atol,
                equal_nan=True,
            )
        )
    flow.clear_default_session()


@flow.unittest.skip_unless_1n1d()
class TestBinaryElementwiseOps(flow.unittest.TestCase):
    def test_floordiv(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.floordiv]
        arg_dict["tf_op"] = [tf.math.floordiv]
        arg_dict["x_shape"] = [(5, 5,)]
        arg_dict["y_shape"] = [(5, 5,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["x_minval"] = [-10]
        arg_dict["x_maxval"] = [10]
        arg_dict["y_minval"] = [1]
        arg_dict["y_maxval"] = [10]
        arg_dict["compare_grad"] = [False]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_pow(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.pow]
        arg_dict["tf_op"] = [tf.math.pow]
        arg_dict["x_shape"] = [(5, 5,)]
        arg_dict["y_shape"] = [(5, 5,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["x_minval"] = [1]
        arg_dict["x_maxval"] = [5]
        arg_dict["y_minval"] = [1]
        arg_dict["y_maxval"] = [5]
        arg_dict["compare_grad"] = [True]

        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_xdivy(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.xdivy]
        arg_dict["tf_op"] = [tf.math.xdivy]
        arg_dict["x_shape"] = [(5, 5,)]
        arg_dict["y_shape"] = [(5, 5,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["x_minval"] = [1]
        arg_dict["x_maxval"] = [100]
        arg_dict["y_minval"] = [1]
        arg_dict["y_maxval"] = [10]
        arg_dict["compare_grad"] = [True]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_xlogy(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.xlogy]
        arg_dict["tf_op"] = [tf.math.xlogy]
        arg_dict["x_shape"] = [(5, 5,)]
        arg_dict["y_shape"] = [(5, 5,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["x_minval"] = [1]
        arg_dict["x_maxval"] = [5]
        arg_dict["y_minval"] = [1]
        arg_dict["y_maxval"] = [5]
        arg_dict["compare_grad"] = [True]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_atan2(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.atan2]
        arg_dict["tf_op"] = [tf.math.atan2]
        arg_dict["x_shape"] = [(5, 5,)]
        arg_dict["y_shape"] = [(5, 5,)]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["x_minval"] = [1]
        arg_dict["x_maxval"] = [5]
        arg_dict["y_minval"] = [1]
        arg_dict["y_maxval"] = [5]
        arg_dict["compare_grad"] = [True]

        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
