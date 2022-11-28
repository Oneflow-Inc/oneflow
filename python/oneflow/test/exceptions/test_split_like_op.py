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


class TestSplitLikeError(flow.unittest.TestCase):
    def test_split_like_like_axes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.randn(4, 4)
            like = (flow.randn(2, 4, 4), flow.randn(2, 4, 4))
            axis = 0
            flow._C.split_like(x, like, axis)
        test_case.assertTrue(
            ") should be less than or equal to input (" in str(context.exception)
        )

    def test_split_like_split_axes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.randn(4, 4)
            like = (flow.randn(2, 4), flow.randn(2, 4))
            axis = 3
            flow._C.split_like(x, like, axis)
        test_case.assertTrue(
            "should be less than the dimension of like" in str(context.exception)
        )

    def test_split_like_like_i_axes_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.randn(4, 4)
            like = (flow.randn(2, 4), flow.randn(2))
            axis = 0
            flow._C.split_like(x, like, axis)
        test_case.assertTrue(
            "must match the dimension of the first like" in str(context.exception)
        )

    def test_split_like_x_i_shape_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.randn(4, 4)
            like = (flow.randn(2, 4), flow.randn(2, 3))
            axis = 0
            flow._C.split_like(x, like, axis)
        test_case.assertTrue("must match the size of like_i" in str(context.exception))

    def test_split_like_non_dynamic_static_dim_err(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.randn(4, 4)
            like = (flow.randn(2, 4), flow.randn(3, 4))
            axis = 0
            flow._C.split_like(x, like, axis)
        test_case.assertTrue(
            "shape situation, the total size of like" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
