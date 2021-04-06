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
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type

import test_global_storage
from test_util import Args, GenArgDict
import oneflow.typing as oft


def compare_with_not_fused(test_case, device_type, x_shape, data_type, data_format):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    if data_format == "NCHW":
        bias_shape = (x_shape[1],)
    elif data_format == "NHWC":
        bias_shape = (x_shape[len(x_shape) - 1],)

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(
        value: oft.Numpy.Placeholder(x_shape), bias: oft.Numpy.Placeholder(bias_shape),
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

            x1 = flow.identity(value)
            x2 = flow.identity(value)

            bias1 = flow.identity(bias)
            bias2 = flow.identity(bias)

            flow.watch_diff(x1, test_global_storage.Setter("x1_diff"))
            flow.watch_diff(x2, test_global_storage.Setter("x2_diff"))

            flow.watch_diff(bias1, test_global_storage.Setter("bias1_diff"))
            flow.watch_diff(bias2, test_global_storage.Setter("bias2_diff"))

            if data_type == "float16":
                y1 = flow.cast(
                    flow.math.gelu(
                        flow.nn.bias_add(
                            flow.cast(x1, dtype=flow.float16),
                            flow.cast(bias1, dtype=flow.float16),
                            data_format=data_format,
                        ),
                    ),
                    dtype=flow.float,
                )
                y2 = flow.cast(
                    flow.nn.fused_bias_add_gelu(
                        flow.cast(x2, dtype=flow.float16),
                        flow.cast(bias2, dtype=flow.float16),
                        data_format=data_format,
                    ),
                    dtype=flow.float,
                )
            else:
                y1 = flow.math.gelu(
                    flow.nn.bias_add(x1, bias1, data_format=data_format)
                )
                y2 = flow.nn.fused_bias_add_gelu(x2, bias2, data_format=data_format)
            flow.watch(y1, test_global_storage.Setter("y1"))
            flow.watch(y2, test_global_storage.Setter("y2"))
            flow.watch_diff(y1, test_global_storage.Setter("y1_diff"))
            flow.watch_diff(y2, test_global_storage.Setter("y2_diff"))

            loss = y1 + y2
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
        ).minimize(flow.math.reduce_sum(loss))

        return loss

    x = np.random.uniform(low=0, high=10, size=x_shape).astype(np.float32)
    bias = np.random.uniform(low=0, high=10, size=bias_shape).astype(np.float32)
    of_out = FlowJob(x, bias).get()

    y1 = test_global_storage.Get("y1")
    y2 = test_global_storage.Get("y2")
    print("y1", y1.flatten()[0:10])
    print("y2", y2.flatten()[0:10])

    tol = 1e-3 if data_type == "float16" else 1e-5
    test_case.assertTrue(np.allclose(y1, y2, rtol=tol, atol=tol, equal_nan=True))
    x1_diff = test_global_storage.Get("x1_diff")
    x2_diff = test_global_storage.Get("x2_diff")
    print("x1_diff", x1_diff.flatten()[0:10])
    print("x2_diff", x2_diff.flatten()[0:10])
    test_case.assertTrue(
        np.allclose(x1_diff, x2_diff, rtol=tol, atol=tol, equal_nan=True)
    )
    bias1_diff = test_global_storage.Get("bias1_diff")
    bias2_diff = test_global_storage.Get("bias2_diff")
    print("bias1_diff", bias1_diff.flatten()[0:10])
    print("bias2_diff", bias2_diff.flatten()[0:10])
    test_case.assertTrue(
        np.allclose(bias1_diff, bias2_diff, rtol=tol, atol=tol, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestFusedBiasAdd(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_fused_bias_add(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [
            (10, 10),
        ]
        arg_dict["data_type"] = ["float16", "float32", "double"]
        arg_dict["data_format"] = ["NCHW"]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            compare_with_not_fused(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
