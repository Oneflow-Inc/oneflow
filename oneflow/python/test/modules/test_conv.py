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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConv2d(flow.unittest.TestCase):
    def test_conv2d(test_case):
        input_arr = np.array(
            [
                [1.1630785, 0.4838046, 0.29956347, 0.15302546, -1.1688148],
                [1.558071, -0.5459446, -2.3556297, 0.54144025, 2.6785064],
                [1.2546344, -0.54877406, -0.68106437, -0.1353156, 0.37723133],
                [0.41016456, 0.5712682, -2.7579627, 1.07628, -0.6141325],
                [1.8307649, -1.1468065, 0.05383794, -2.5074806, -0.5916499],
            ]
        ).reshape((1, 1, 5, 5))
        output_arr = np.array(
            [
                [1.4438007, -6.4144573, -0.66943216],
                [-7.1190453, -11.122115, -4.302488],
                [-2.3320563, -13.974841, -13.29459],
                [1.4438007, -6.4144573, -0.66943216],
                [-7.1190453, -11.122115, -4.302488],
                [-2.3320563, -13.974841, -13.29459],
                [1.4438007, -6.4144573, -0.66943216],
                [-7.1190453, -11.122115, -4.302488],
                [-2.3320563, -13.974841, -13.29459],
            ]
        ).reshape((1, 3, 3, 3))
        conv = flow.nn.Conv2d(1, 3, (3, 3), bias=False)
        x = flow.Tensor(input_arr)
        flow.nn.init.constant_(conv.weight, 2.3)
        of_out = conv(x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), output_arr, rtol=1e-2, atol=1e-5)
        )

    def test_conv2d_with_bias(test_case):
        input_arr = np.array(
            [
                [1.1630785, 0.4838046, 0.29956347, 0.15302546, -1.1688148],
                [1.558071, -0.5459446, -2.3556297, 0.54144025, 2.6785064],
                [1.2546344, -0.54877406, -0.68106437, -0.1353156, 0.37723133],
                [0.41016456, 0.5712682, -2.7579627, 1.07628, -0.6141325],
                [1.8307649, -1.1468065, 0.05383794, -2.5074806, -0.5916499],
            ]
        ).reshape((1, 1, 5, 5))
        output_arr = np.array(
            [
                [3.7438006, -4.114457, 1.6305678],
                [-4.819045, -8.822115, -2.002488],
                [-0.03205633, -11.674841, -10.99459],
                [3.7438006, -4.114457, 1.6305678],
                [-4.819045, -8.822115, -2.002488],
                [-0.03205633, -11.674841, -10.99459],
                [3.7438006, -4.114457, 1.6305678],
                [-4.819045, -8.822115, -2.002488],
                [-0.03205633, -11.674841, -10.99459],
            ]
        ).reshape((1, 3, 3, 3))
        conv = flow.nn.Conv2d(1, 3, (3, 3))
        x = flow.Tensor(input_arr)
        flow.nn.init.constant_(conv.weight, 2.3)
        flow.nn.init.constant_(conv.bias, 2.3)
        of_out = conv(x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), output_arr, rtol=1e-2, atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
