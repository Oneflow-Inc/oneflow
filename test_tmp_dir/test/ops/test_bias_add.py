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
from test_util import Args, GenArgDict
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowBiasAdd(data_type, device_type, value, bias, flow_args):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(
        value: oft.Numpy.Placeholder(value.shape),
        bias: oft.Numpy.Placeholder(bias.shape),
    ):
        with flow.scope.placement(device_type, "0:0"):
            value += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            bias += flow.get_variable(
                name="v2",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            if data_type == "float16":
                comp_value = flow.cast(value, dtype=flow.float16)
                comp_bias = flow.cast(bias, dtype=flow.float16)
            else:
                comp_value = value
                comp_bias = bias
            loss = flow.nn.bias_add(comp_value, comp_bias, *flow_args)
            if data_type == "float16":
                loss = flow.cast(loss, dtype=flow.float)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
            ).minimize(loss)

            flow.watch_diff(value, test_global_storage.Setter("value_diff"))
            flow.watch_diff(bias, test_global_storage.Setter("bias_diff"))

            return loss

    # OneFlow
    y = FlowJob(value, bias).get().numpy()
    value_diff = test_global_storage.Get("value_diff")
    bias_diff = test_global_storage.Get("bias_diff")
    return y, value_diff, bias_diff


def RunTensorFlowBiasAdd(data_type, value, bias, tf_args):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        value, bias = tf.Variable(value), tf.Variable(bias)
        if data_type == "float16":
            comp_value = tf.cast(value, tf.float16)
            comp_bias = tf.cast(bias, tf.float16)
        else:
            comp_value = value
            comp_bias = bias
        y = tf.nn.bias_add(comp_value, comp_bias, *tf_args)
    value_diff = tape.gradient(y, value).numpy()
    bias_diff = tape.gradient(y, bias).numpy()
    return y.numpy(), value_diff, bias_diff


def CompareBiasAddWithTensorFlow(
    data_type,
    device_type,
    input_shapes,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = [
        np.random.uniform(low=input_minval, high=input_maxval, size=input_shape).astype(
            np.float32
        )
        for input_shape in input_shapes
    ]
    of_y, of_x_diff1, of_x_diff2 = RunOneflowBiasAdd(
        data_type, device_type, *x, flow_args
    )
    tf_y, tf_x_diff1, tf_x_diff2 = RunTensorFlowBiasAdd(data_type, *x, tf_args)
    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(of_x_diff1, tf_x_diff1, rtol=x_diff_rtol, atol=x_diff_atol)
    assert np.allclose(of_x_diff2, tf_x_diff2, rtol=x_diff_rtol, atol=x_diff_atol)


@flow.unittest.skip_unless_1n1d()
class TestBiasAdd(flow.unittest.TestCase):
    def test_bias_add_nchw(test_case):
        arg_dict = OrderedDict()
        arg_dict["data_type"] = ["float16", "float32"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shapes"] = [((1, 20, 1, 11), (20,)), ((2, 20, 1, 11), (20,))]
        arg_dict["op_args"] = [Args(["NCHW"])]
        for arg in GenArgDict(arg_dict):
            if arg["data_type"] == "float16" and arg["device_type"] == "cpu":
                continue
            CompareBiasAddWithTensorFlow(**arg)

    def test_bias_add_nhwc(test_case):
        arg_dict = OrderedDict()
        arg_dict["data_type"] = ["float16", "float32"]
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shapes"] = [((30, 20, 5, 10), (10,)), ((2, 5, 7, 7), (7,))]
        arg_dict["op_args"] = [Args(["NHWC"])]
        for arg in GenArgDict(arg_dict):
            if arg["data_type"] == "float16" and arg["device_type"] == "cpu":
                continue
            CompareBiasAddWithTensorFlow(**arg)


if __name__ == "__main__":
    unittest.main()
