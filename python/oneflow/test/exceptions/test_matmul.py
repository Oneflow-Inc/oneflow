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


class TestMatmulError(flow.unittest.TestCase):
    def test_matmul_dimension_error1(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4,), dtype=flow.float32)
            w = flow.ones((4, 4), dtype=flow.float32)
            out = flow._C.matmul(x, w, False, False, 1.0)
        test_case.assertTrue(
            "Check failed: (a_shape->NumAxes()) >= (2) (1 vs 2) Tensor a's dim should >= 2"
            in str(context.exception)
        )

    def test_matmul_dimension_error2(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32)
            w = flow.ones((4,), dtype=flow.float32)
            out = flow._C.matmul(x, w, False, False, 1.0)
        test_case.assertTrue(
            "Check failed: (b_shape->NumAxes()) >= (2) (1 vs 2) Tensor b's dim should >= 2"
            in str(context.exception)
        )

    def test_matmul_dimension_error3(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 1, 2, 1), dtype=flow.float32)
            w = flow.ones((4, 4, 4), dtype=flow.float32)
            out = flow._C.matmul(x, w, False, False, 1.0)

        test_case.assertTrue(
            "Check failed: (b_shape->NumAxes()) == (2) (3 vs 2) Not support number of dimensions of a being less than number of dimensions of b!"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
