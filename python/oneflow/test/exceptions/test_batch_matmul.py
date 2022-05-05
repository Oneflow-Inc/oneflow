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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestBatchMatmulError(flow.unittest.TestCase):
    def test_batch_matmul_dimension_error1(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 4), dtype=flow.float32)
            w = flow.ones((4, 1, 1), dtype=flow.float32)
            out = flow._C.batch_matmul(x, w, False, False, 1.0)

        test_case.assertTrue(
            "Check failed: (a_shape->NumAxes()) >= (3) (2 vs 3) Tensor a's dim should >= 3"
            in str(ctx.exception)
        )

    def test_batch_matmul_dimension_error2(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 1, 1), dtype=flow.float32)
            w = flow.ones((4,), dtype=flow.float32)
            out = flow._C.batch_matmul(x, w, False, False, 1.0)

        test_case.assertTrue(
            "Check failed: (b_shape->NumAxes()) >= (3) (1 vs 3) Tensor b's dim should >= 3"
            in str(ctx.exception)
        )

    def test_batch_matmul_batch_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 1, 1), dtype=flow.float32)
            w = flow.ones((2, 1, 1), dtype=flow.float32)
            out = flow._C.batch_matmul(x, w, False, False, 1.0)

        test_case.assertTrue(
            " Check failed: (a_shape->At(0)) == (b_shape->At(0)) (4 vs 2) batch dim not match, please check input!"
            in str(ctx.exception)
        )

    def test_batch_matmul_matrix_dim_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((4, 2, 3), dtype=flow.float32)
            w = flow.ones((4, 1, 2), dtype=flow.float32)
            out = flow._C.batch_matmul(x, w, False, False, 1.0)

        test_case.assertTrue(
            "Check failed: (a_shape->At(2)) == (b_shape->At(1)) (3 vs 1) matmul dim not match, please check input!"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
