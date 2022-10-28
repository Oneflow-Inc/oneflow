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


@flow.unittest.skip_unless_1n1d()
class TestMv(flow.unittest.TestCase):
    def test_mv_not_matrix(test_case):
        with test_case.assertRaises(Exception) as exp:
            mat = flow.randn(2, 3, 3)
            vec = flow.randn(3)
            out = flow.mv(mat, vec)
        test_case.assertTrue(
            "vector + matrix @ vector expected, got 1, 3, 1" in str(exp.exception)
        )

    def test_mv_not_vector(test_case):
        with test_case.assertRaises(Exception) as exp:
            mat = flow.randn(2, 3)
            vec = flow.randn(3, 1)
            out = flow.mv(mat, vec)
        test_case.assertTrue(
            "vector + matrix @ vector expected, got 1, 2, 2" in str(exp.exception)
        )

    def test_mv_size_mismatch(test_case):
        with test_case.assertRaises(Exception) as exp:
            mat = flow.randn(2, 3)
            vec = flow.randn(4)
            out = flow.mv(mat, vec)
        test_case.assertTrue("size mismatch" in str(exp.exception))


if __name__ == "__main__":
    unittest.main()
