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
from test_util import (
    GenArgDict,
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowOp(device_type, flow_op, x, y, data_type):
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


def RunTensorFlowOp(tf_op, x, y):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = tf.Variable(y)
        out = tf_op(x, y)
    x_diff = tape.gradient(out, x)
    y_diff = tape.gradient(out, y)
    return out.numpy(), x_diff.numpy(), y_diff.numpy()


def compare_with_tensorflow_grad(
    device_type,
    flow_op,
    tf_op,
    x_shape,
    y_shape,
    data_type,
    input_minval=-10,
    input_maxval=10,
    out_rtol=1e-5,
    out_atol=1e-5,
    diff_rtol=1e-4,
    diff_atol=1e-3,
):
    assert device_type in ["gpu", "cpu"]

    np_type = type_name_to_np_type[data_type]
    x = np.random.uniform(low=input_minval, high=input_maxval, size=x_shape).astype(
        np_type
    )
    y = np.random.uniform(low=input_minval, high=input_maxval, size=y_shape).astype(
        np_type
    )
    if flow_op in (flow.math.divide, flow.math.mod):
        y[np.where(y == 0)] += 1

    of_out, of_x_diff, of_y_diff, = RunOneflowOp(device_type, flow_op, x, y, data_type)
    tf_out, tf_x_diff, tf_y_diff = RunTensorFlowOp(tf_op, x, y)

    assert np.allclose(of_out, tf_out, rtol=out_rtol, atol=out_atol, equal_nan=True)
    assert np.allclose(
        of_x_diff, tf_x_diff, rtol=diff_rtol, atol=diff_atol, equal_nan=True
    )
    assert np.allclose(
        of_y_diff, tf_y_diff, rtol=diff_rtol, atol=diff_atol, equal_nan=True
    )
    flow.clear_default_session()


def compare_with_tensorflow(
    device_type,
    flow_op,
    tf_op,
    x_shape,
    y_shape,
    data_type,
    input_minval=-10,
    input_maxval=10,
    out_rtol=1e-5,
    out_atol=1e-5,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    flow_type = type_name_to_flow_type[data_type]

    @flow.global_function(function_config=func_config)
    def FlowJob(
        x: oft.Numpy.Placeholder(x_shape, dtype=flow_type),
        y: oft.Numpy.Placeholder(y_shape, dtype=flow_type),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow_op(x, y)

    np_type = type_name_to_np_type[data_type]
    if np_type in (np.int8, np.int32, np.int64):
        x = np.random.randint(low=input_minval, high=input_maxval, size=x_shape).astype(
            np_type
        )
        y = np.random.randint(low=input_minval, high=input_maxval, size=y_shape).astype(
            np_type
        )
    else:
        x = np.random.uniform(low=input_minval, high=input_maxval, size=x_shape).astype(
            np_type
        )
        y = np.random.uniform(low=input_minval, high=input_maxval, size=y_shape).astype(
            np_type
        )
    if flow_op in (flow.math.divide, flow.math.mod):
        y[np.where(y == 0)] += 1

    # Oneflow
    of_out = FlowJob(x, y).get().numpy()
    # Tensorflow
    tf_out = tf_op(x, y).numpy()
    assert np.allclose(of_out, tf_out, rtol=out_rtol, atol=out_atol, equal_nan=True)
    flow.clear_default_session()


@flow.unittest.skip_unless_1n1d()
class TestBroadcastNormal(flow.unittest.TestCase):
    def test_broadcast_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["flow_op"] = [flow.math.add]
        arg_dict["tf_op"] = [tf.math.add]
        arg_dict["x_shape"] = [(3, 1, 4, 1)]
        arg_dict["y_shape"] = [(4, 1, 6)]
        arg_dict["data_type"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow_grad(*arg)

    def test_broadcast_sub(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["flow_op"] = [flow.math.subtract]
        arg_dict["tf_op"] = [tf.math.subtract]
        arg_dict["x_shape"] = [(3, 1, 4, 1)]
        arg_dict["y_shape"] = [(4, 1, 6)]
        arg_dict["data_type"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_broadcast_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.multiply]
        arg_dict["tf_op"] = [tf.math.multiply]
        arg_dict["x_shape"] = [(3, 1, 4, 5, 1)]
        arg_dict["y_shape"] = [(1, 4, 1, 1, 5)]
        arg_dict["data_type"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow_grad(*arg)

    def test_broadcast_div(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["flow_op"] = [flow.math.divide]
        arg_dict["tf_op"] = [tf.math.divide]
        arg_dict["x_shape"] = [(3, 1, 4, 5, 1)]
        arg_dict["y_shape"] = [(3, 4, 1, 1, 5)]
        arg_dict["data_type"] = ["float32", "double"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow_grad(*arg)

    def test_broadcast_floormod(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["flow_op"] = [flow.math.mod]
        arg_dict["tf_op"] = [tf.math.floormod]
        arg_dict["x_shape"] = [(3, 1, 4, 5, 1)]
        arg_dict["y_shape"] = [(1, 4, 1, 1, 5)]
        arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_broadcast_maximum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["flow_op"] = [flow.math.maximum]
        arg_dict["tf_op"] = [tf.math.maximum]
        arg_dict["x_shape"] = [(3, 1, 4, 5, 1)]
        arg_dict["y_shape"] = [(1, 4, 1, 1, 5)]
        arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_broadcast_minimum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["flow_op"] = [flow.math.minimum]
        arg_dict["tf_op"] = [tf.math.minimum]
        arg_dict["x_shape"] = [(3, 1, 4, 5, 1)]
        arg_dict["y_shape"] = [(1, 4, 1, 1, 5)]
        arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
