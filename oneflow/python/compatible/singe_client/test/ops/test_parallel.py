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


def NaiveTest(test_case):
    shape = (16, 2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def AddJob(a: oft.Numpy.Placeholder(shape), b: oft.Numpy.Placeholder(shape)):
        return a + b + b

    x = np.random.rand(*shape).astype(np.float32)
    y = np.random.rand(*shape).astype(np.float32)
    z = AddJob(x, y).get().numpy()
    test_case.assertTrue(np.array_equal(z, x + y + y))


class TestParallel(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_1n1c(test_case):
        flow.config.gpu_device_num(1)
        NaiveTest(test_case)

    @flow.unittest.skip_unless_1n2d()
    def test_1n2c(test_case):
        flow.config.gpu_device_num(2)
        NaiveTest(test_case)

    @flow.unittest.skip_unless_2n1d()
    def test_2n2c(test_case):
        flow.config.gpu_device_num(1)
        NaiveTest(test_case)


if __name__ == "__main__":
    unittest.main()
