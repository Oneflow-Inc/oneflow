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
import oneflow as flow
import unittest
import numpy as np


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMathModule(flow.unittest.TestCase):
    def test_sin(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = flow.sin(input)
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
        test_case.assertTrue(np.allclose(input.sin().numpy(), np_out, 1e-5, 1e-5))

        arr = np.array([-0.5461, 0.1347, -2.7266, -0.2746])
        input2 = flow.Tensor(arr, dtype=flow.float32)
        np_out2 = np.array([-0.51935846, 0.13429303, -0.40318328, -0.27116194])
        of_out2 = flow.sin(input2)
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out2, 1e-5, 1e-5))

    def test_cos(test_case):
        input = flow.Tensor(np.random.randn(1, 3, 6), dtype=flow.float32)
        of_out = flow.cos(input)
        np_out = np.cos(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
        test_case.assertTrue(np.allclose(input.cos().numpy(), np_out, 1e-5, 1e-5))

        arr = np.array([1.4309, 1.2706, -0.8562, 0.9796])
        input2 = flow.Tensor(arr, dtype=flow.float32)
        np_out2 = np.array([0.13944048, 0.29570782, 0.6553126, 0.5573547])
        of_out2 = flow.cos(input2)
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out2))

    def test_log(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        of_out = flow.log(input)
        np_out = np.log(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))
        test_case.assertTrue(np.allclose(input.log().numpy(), np_out, equal_nan=True))

        arr = np.array([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
        input2 = flow.Tensor(arr, dtype=flow.float32)
        np_out = np.full((5,), np.nan)
        of_out2 = flow.log(input2)
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
