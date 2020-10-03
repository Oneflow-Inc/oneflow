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


@flow.unittest.skip_unless_1n2d()
class TestWatch(flow.unittest.TestCase):
    def test_simple(test_case):
        flow.config.gpu_device_num(1)
        data = np.ones((10,), dtype=np.float32)

        def EqOnes(x):
            test_case.assertTrue(np.allclose(data, x.numpy()))

        @flow.global_function()
        def ReluJob(x: oft.Numpy.Placeholder((10,))):
            flow.watch(x, EqOnes)

        ReluJob(data)

    def test_two_device(test_case):
        flow.config.gpu_device_num(2)
        data = np.ones((10,), dtype=np.float32)

        def EqOnes(x):
            test_case.assertTrue(np.allclose(data, x.numpy()))

        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def ReluJob(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.relu(x)
            flow.watch(y, EqOnes)

        ReluJob(data)


if __name__ == "__main__":
    unittest.main()
