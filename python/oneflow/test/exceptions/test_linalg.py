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
import oneflow.unittest


class TestLinalgCross(flow.unittest.TestCase):
    def test_cross_has_no_3_error(test_case):
        a = flow.randn(4, 2)
        b = flow.randn(4, 2)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.cross(a, b)
            test_case.assertTrue(
                "RuntimeError: no dimension of size 3 in input." in str(ctx.exception)
            )

    def test_linalg_cross_has_no_3_error(test_case):
        a = flow.randn(4, 2)
        b = flow.randn(4, 2)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.linalg.cross(a, b)
            test_case.assertTrue(
                "RuntimeError: the size of the specified dimension(which is -1) is not 3."
                in str(ctx.exception)
            )

    def test_linalg_cross_broadcast_error(test_case):
        a = flow.randn(4)
        b = flow.randn(4, 2)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.linalg.cross(a, b)
            test_case.assertTrue(
                "RuntimeError: input and other can't be broadcasted to a single shape. [input's shape: (1,4), other's shape: (4,2)]."
                in str(ctx.exception)
            )


if __name__ == "__main__":
    unittest.main()
