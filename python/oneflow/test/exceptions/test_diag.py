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


@flow.unittest.skip_unless_1n1d()
class TestDiag(flow.unittest.TestCase):
    def test_diag_dims_error(test_case):
        with test_case.assertRaises(RuntimeError) as exp:
            a = flow.randn(2, 3, 4)
            a.diag(diagonal=0)
        test_case.assertTrue(
            "expected to be in range of [1, 2], but got 3" in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()
