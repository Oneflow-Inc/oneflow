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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestWhere(flow.unittest.TestCase):
    def test_where(test_case):
        x = flow.Tensor(
            np.array([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
            dtype=flow.float32,
        )
        y = flow.Tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        condition = flow.Tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        of_out = flow.where(condition, x, y)
        np_out = np.array([[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_tensor_where(test_case):
        x = flow.Tensor(
            np.array([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
            dtype=flow.float32,
        )
        y = flow.Tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        condition = flow.Tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        of_out = condition.where(x, y)
        np_out = np.array([[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_where_broadcast(test_case):
        x = flow.Tensor(
            np.array([[[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
            dtype=flow.float32,
        )
        y = flow.Tensor(np.ones(shape=(3, 3, 2)), dtype=flow.float32)
        condition = flow.Tensor(np.array([[[0, 1], [1, 0], [1, 0]]]), dtype=flow.int32)
        of_out = flow.where(condition, x, y)
        np_out = np.array(
            [
                [[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]],
                [[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]],
                [[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]],
            ]
        )
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_where_scalar(test_case):
        x = 0.5
        y = 2.0
        condition = flow.Tensor(np.array([1]), dtype=flow.int32)
        of_out = flow.where(condition, x, y)
        np_out = np.array([0.5])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_where_dim4(test_case):
        x = flow.Tensor(
            np.array([[[[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]]),
            dtype=flow.float32,
        )
        y = flow.Tensor(np.ones(shape=(1, 1, 3, 2)), dtype=flow.float32)
        condition = flow.Tensor(
            np.array([[[[0, 1], [1, 0], [1, 0]]]]), dtype=flow.int32
        )
        of_out = flow.where(condition, x, y)
        np_out = np.array([[[[1.0000, 0.3139], [0.3898, 1.0000], [0.0478, 1.0000]]]])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
