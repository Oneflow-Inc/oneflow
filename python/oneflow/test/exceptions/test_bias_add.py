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


class TestBiasAddError(flow.unittest.TestCase):
    def test_bias_add_dimension_match_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((5,), dtype=flow.float32)
            out = flow._C.bias_add(x, bias, axis=1)

        test_case.assertTrue(
            "The size of tensor x (4,4) must match the size of tensor b (5,) at dimension 1"
            in str(context.exception)
        )

    def test_bias_add_index_error(test_case):
        with test_case.assertRaises(Exception) as context:
            x = flow.ones((4, 4), dtype=flow.float32)
            bias = flow.ones((5,), dtype=flow.float32)
            out = flow._C.bias_add(x, bias, axis=3)

        test_case.assertTrue(
            "Dimension out of range (expected to be in range of [ -2,1], but got 3)"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
