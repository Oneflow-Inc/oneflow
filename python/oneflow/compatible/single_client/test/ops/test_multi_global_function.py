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

import os
import unittest

import numpy as np

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


def ccrelu(x, name):
    return (
        flow.user_op_builder(name)
        .Op("ccrelu")
        .Input("x", [x])
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fixed_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ReluJob1(a: oft.Numpy.Placeholder((5, 2))):
        return ccrelu(a, "my_cc_relu_op_1")

    @flow.global_function(function_config=func_config)
    def ReluJob2(a: oft.Numpy.Placeholder((10, 2))):
        return ccrelu(a, "my_cc_relu_op_2")

    x = np.random.rand(5, 2).astype(np.float32)
    y = ReluJob1(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))

    x = np.random.rand(10, 2).astype(np.float32)
    y = ReluJob2(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


class TestMultiGlobalFunction(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_multi_global_function(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)


if __name__ == "__main__":
    unittest.main()
