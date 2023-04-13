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
import oneflow.nn.functional as F


@flow.unittest.skip_unless_1n1d()
class TestMm(flow.unittest.TestCase):
    def test_mm_not_2dim(test_case):
        with test_case.assertRaises(Exception) as exp:
            mat1 = flow.randn(2, 3, 3)
            mat2 = flow.randn(3, 3)
            out = flow.mm(mat1, mat2)
        test_case.assertTrue("self must be a matrix" in str(exp.exception))
        with test_case.assertRaises(Exception) as exp:
            mat1 = flow.randn(2, 3)
            mat2 = flow.randn(3, 3, 2)
            out = flow.mm(mat1, mat2)
        test_case.assertTrue("mat2 must be a matrix" in str(exp.exception))

    def test_mm_dim_not_match(test_case):
        with test_case.assertRaises(Exception) as exp:
            mat1 = flow.randn(2, 3)
            mat2 = flow.randn(4, 3)
            out = flow.mm(mat1, mat2)
        test_case.assertTrue(
            "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x3)"
            in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()
