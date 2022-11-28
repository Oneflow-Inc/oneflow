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
class TestInv(flow.unittest.TestCase):
    def test_inv_exception_dim_short(test_case):
        x = flow.tensor((2, 2))
        with test_case.assertRaises(RuntimeError) as ctx:
            y = flow.linalg.inv(x)
        test_case.assertTrue(
            "linalg.inv: The input tensor must be at least 2 dimensions."
            in str(ctx.exception)
        )

    def test_inv_exception_not_square_matrix(test_case):
        x = flow.randn(2, 3, 2)
        with test_case.assertRaises(RuntimeError) as ctx:
            y = flow.linalg.inv(x)
        test_case.assertTrue(
            "RuntimeError: linalg.inv: A must be batches of square matrices, but they are 3 by 2 matrices"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
