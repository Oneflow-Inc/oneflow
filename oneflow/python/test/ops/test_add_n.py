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
import oneflow.typing as oft
from typing import Tuple

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def GenerateTest(test_case, shape, num_inputs):
    @flow.global_function(function_config=func_config)
    def AddJob(xs: Tuple[(oft.Numpy.Placeholder(shape),) * num_inputs]):
        return flow.math.add_n(xs)

    inputs = tuple(np.random.rand(*shape).astype(np.float32) for i in range(num_inputs))
    r = AddJob(inputs).get().numpy()
    test_case.assertTrue(np.allclose(r, sum(inputs)))


@flow.unittest.skip_unless_1n1d()
class TestAddN(flow.unittest.TestCase):
    def test_naive(test_case):
        @flow.global_function(function_config=func_config)
        def AddJob(xs: Tuple[(oft.Numpy.Placeholder((5, 2)),) * 3]):
            return flow.math.add_n(xs)

        inputs = tuple(np.random.rand(5, 2).astype(np.float32) for i in range(3))
        r = AddJob(inputs).get().numpy()
        test_case.assertTrue(np.allclose(r, sum(inputs)))

    def test_2_inputs(test_case):
        GenerateTest(test_case, (64, 64), 2)

    def test_3_inputs(test_case):
        GenerateTest(test_case, (64, 64), 3)

    def test_4_inputs(test_case):
        GenerateTest(test_case, (64, 64), 4)

    def test_5_inputs(test_case):
        GenerateTest(test_case, (64, 64), 5)

    def test_6_inputs(test_case):
        GenerateTest(test_case, (64, 64), 6)

    def test_7_inputs(test_case):
        GenerateTest(test_case, (64, 64), 7)

    def test_8_inputs(test_case):
        GenerateTest(test_case, (64, 64), 8)

    def test_9_inputs(test_case):
        GenerateTest(test_case, (64, 64), 9)

    def test_10_inputs(test_case):
        GenerateTest(test_case, (64, 64), 10)

    def test_11_inputs(test_case):
        GenerateTest(test_case, (64, 64), 11)

    def test_12_inputs(test_case):
        GenerateTest(test_case, (64, 64), 12)

    def test_13_inputs(test_case):
        GenerateTest(test_case, (64, 64), 13)

    def test_14_inputs(test_case):
        GenerateTest(test_case, (64, 64), 14)

    def test_15_inputs(test_case):
        GenerateTest(test_case, (64, 64), 15)

    def test_16_inputs(test_case):
        GenerateTest(test_case, (64, 64), 16)

    def test_100_inputs(test_case):
        GenerateTest(test_case, (64, 64), 100)


if __name__ == "__main__":
    unittest.main()
