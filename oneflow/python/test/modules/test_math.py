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
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):

    def test_sin(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = flow.sin(input)
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))
    
    def test_cos(test_case):
        input = flow.Tensor(np.random.randn(1, 3, 6), dtype=flow.float32)
        of_out = flow.cos(input)
        np_out = np.cos(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_log(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        of_out = flow.log(input)
        np_out = np.log(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, equal_nan=True))

    def test_std(test_case):
        np_arr = np.array(
            [[-0.39283446,  0.44999730,  0.25533655],
            [ 0.76964611,  0.17798263,  1.46315704]]
        )
        input = flow.Tensor(np_arr)
        of_out = flow.npstd(input, 1)
        np_out = np.std(np_arr, axis=1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        np_arr2 = np.random.randn(2, 3, 4, 5)
        input2 = flow.Tensor(np_arr2)
        of_out2 = flow.npstd(input2, 2)
        np_out2 = np.std(np_arr2, axis=2)
        test_case.assertTrue(np.allclose(of_out2.numpy(), np_out2, 1e-5, 1e-5))


if __name__ == "__main__":
    unittest.main()
