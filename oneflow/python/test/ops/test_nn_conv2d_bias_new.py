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
import numpy as np
import oneflow as flow
import test_global_storage
from collections import OrderedDict
from test_util import GenArgList


def compare(
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
                initializer=flow.zeros_initializer(),
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
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
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

    of_bias_out = RunConvBias()
    flow.clear_default_session()

    @flow.global_function(type="train", function_config=func_config)
    def RunConv():
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
            loss = flow.nn.conv2d(
                x,
                weight,
                strides=[stride, stride],
                padding=padding,
                dilations=[1, 1],
                groups=groups,
                name="conv",
            )

            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("_x"))
            flow.watch_diff(x, test_global_storage.Setter("_x_diff"))
            flow.watch(weight, test_global_storage.Setter("_weight"))
            flow.watch_diff(weight, test_global_storage.Setter("_weight_diff"))
            flow.watch(loss, test_global_storage.Setter("_loss"))
            flow.watch_diff(loss, test_global_storage.Setter("_loss_diff"))

            return loss

    of_out = RunConv()
    flow.clear_default_session()
    assert np.allclose(of_bias_out.numpy(), of_out.numpy(), rtol=5e-3, atol=5e-3)


@flow.unittest.skip_unless_1n1d()
class TestNnConv2dBias(flow.unittest.TestCase):
    def test_cpu_group1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["x_shape"] = [(3, 32, 128, 128)]
        arg_dict["filters"] = [5]
        arg_dict["kernel_size"] = [1]
        arg_dict["groups"] = [1]
        for arg in GenArgList(arg_dict):
            compare(*arg)


if __name__ == "__main__":
    unittest.main()
