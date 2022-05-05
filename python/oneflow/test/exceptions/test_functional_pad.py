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


class TestPadError(flow.unittest.TestCase):
    def test_pad_ndim_limit_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1, 1, 1, 1, 1), dtype=flow.float32)
            out = flow._C.pad(x, (1,))

        test_case.assertTrue(
            "Check failed: (ndim) <= (5) (6 vs 5) Dimension of input tensor should less than or equal to 5"
            in str(ctx.exception)
        )

    def test_pad_size_attribute_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1), dtype=flow.float32)
            out = flow._C.pad(x, (1, 1, 1, 1, 1))
        test_case.assertTrue(
            "Check failed: (pad.size()) <= (2 * ndim) (5 vs 4) Pad size should less than or equal to input axes * 2."
            in str(ctx.exception)
        )

    def test_pad_size_mod2_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1), dtype=flow.float32)
            out = flow._C.pad(x, (1, 1, 1,))

        test_case.assertTrue(
            "Check failed: (pad.size() % 2) == (0) (1 vs 0) Length of pad must be even but instead it equals 3"
            in str(ctx.exception)
        )

    def test_reflect_pad_size_error(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((1, 1, 2, 2), dtype=flow.float32)
            out = flow._C.pad(x, (4, 4, 4, 4), mode="reflect")

        test_case.assertTrue(
            "Check failed: pad[2] < pad_h && pad[3] < pad_h && pad[0] < pad_w && pad[1] < pad_w padding size should be less than the corresponding input dimension!"
            in str(ctx.exception)
        )

    def test_pad_mode_error(test_case):
        with test_case.assertRaises(NotImplementedError) as ctx:
            x = flow.ones((1, 1, 2, 2), dtype=flow.float32)
            out = flow._C.pad(x, (4, 4, 4, 4), mode="test")

        test_case.assertTrue(
            "Pad mode is test, but only constant, reflect and replicate are valid."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
