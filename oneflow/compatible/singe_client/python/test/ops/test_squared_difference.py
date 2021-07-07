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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.typing as oft
import tensorflow as tf
from test_util import Args, CompareOpWithTensorFlow, GenArgDict

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def GenerateTest(test_case, a_shape, b_shape):
    @flow.global_function(function_config=func_config)
    def SqrDiffJob(
        a: oft.Numpy.Placeholder(a_shape), b: oft.Numpy.Placeholder(b_shape)
    ):
        return flow.math.squared_difference(a, b)

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)
    y = SqrDiffJob(a, b).get().numpy()
    test_case.assertTrue(np.allclose(y, (a - b) * (a - b)))


@flow.unittest.skip_unless_1n1d()
class TestSquaredDifference(flow.unittest.TestCase):
    def test_naive(test_case):
        @flow.global_function(function_config=func_config)
        def SqrDiffJob(
            a: oft.Numpy.Placeholder((5, 2)), b: oft.Numpy.Placeholder((5, 2))
        ):
            return flow.math.squared_difference(a, b)

        x = np.random.rand(5, 2).astype(np.float32)
        y = np.random.rand(5, 2).astype(np.float32)
        z = None
        z = SqrDiffJob(x, y).get().numpy()
        test_case.assertTrue(np.allclose(z, (x - y) * (x - y)))

    def test_broadcast(test_case):
        @flow.global_function(function_config=func_config)
        def SqrDiffJob(
            a: oft.Numpy.Placeholder((5, 2)), b: oft.Numpy.Placeholder((1, 2))
        ):
            return flow.math.squared_difference(a, b)

        x = np.random.rand(5, 2).astype(np.float32)
        y = np.random.rand(1, 2).astype(np.float32)
        z = None
        z = SqrDiffJob(x, y).get().numpy()
        test_case.assertTrue(np.allclose(z, (x - y) * (x - y)))

    def test_xy_sqr_diff_x1(test_case):
        GenerateTest(test_case, (64, 64), (64, 1))

    def test_xy_sqr_diff_1y(test_case):
        GenerateTest(test_case, (64, 64), (1, 64))

    def test_xyz_sqr_diff_x1z(test_case):
        GenerateTest(test_case, (64, 64, 64), (64, 1, 64))

    def test_xyz_sqr_diff_1y1(test_case):
        GenerateTest(test_case, (64, 64, 64), (1, 64, 1))

    def test_scalar_sqr_diff(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["flow_op"] = [flow.math.squared_difference]
        arg_dict["tf_op"] = [tf.math.squared_difference]
        arg_dict["input_shape"] = [(10, 10, 10)]
        arg_dict["op_args"] = [
            Args([1]),
            Args([-1]),
            Args([84223.19348]),
            Args([-3284.139]),
        ]
        for arg in GenArgDict(arg_dict):
            CompareOpWithTensorFlow(**arg)


if __name__ == "__main__":
    unittest.main()
