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


@flow.unittest.skip_unless_1n1d()
class TestScalarDivide(flow.unittest.TestCase):
    def test_scalar_div_2(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        func_config.default_data_type(flow.float)

        @flow.global_function(function_config=func_config)
        def Div2Job(a: oft.Numpy.Placeholder((10, 10))):
            return a / 2

        x = np.random.rand(10, 10).astype(np.float32) + 1
        y = Div2Job(x).get().numpy()
        test_case.assertTrue(np.allclose(y, x / 2))

    def test_scalar_div_by_2(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        func_config.default_data_type(flow.float)

        @flow.global_function(function_config=func_config)
        def DivBy2Job(a: oft.Numpy.Placeholder((10, 10))):
            return 2 / a

        x = np.random.rand(10, 10).astype(np.float32) + 1
        y = DivBy2Job(x).get().numpy()
        test_case.assertTrue(np.allclose(y, 2 / x))


if __name__ == "__main__":
    unittest.main()
