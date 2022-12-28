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


class TestReduceSumLikeOps(flow.unittest.TestCase):
    def test_reduce_sum_like_empty_axis_case_err(test_case):
        a = flow.tensor([1, 1])
        b = flow.tensor([1, 1, 1])
        with test_case.assertRaises(RuntimeError) as ctx:
            flow._C.reduce_sum_like(a, b, [])
        test_case.assertTrue(
            "The shape of the x tensor must be consistent to the shape of the like tensor"
            in str(ctx.exception)
        )

    def test_reduce_sum_like_type_err(test_case):
        a = flow.tensor([1, 1], dtype=flow.int64)
        b = flow.tensor([1, 1], dtype=flow.float64)
        with test_case.assertRaises(TypeError) as ctx:
            flow._C.reduce_sum_like(a, b, [1])
        test_case.assertTrue(
            "Tensors x and like must have the same type" in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
