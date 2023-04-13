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


class TestRoiAlignOp(flow.unittest.TestCase):
    def test_rol_align_x_tensor_dimension_err(test_case):
        x = flow.randn(2, 3, 64)
        rois = flow.randn(2, 3, 64, 64)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.roi_align(x, rois, 2.0, 14, 14, 2, True)
        test_case.assertTrue(
            "The dimension of x tensor must be equal to 4, but got"
            in str(ctx.exception)
        )

    def test_rol_align_rois_tensor_dimension_err(test_case):
        x = flow.randn(2, 3, 64, 5)
        rois = flow.randn(2, 3, 64, 64)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.roi_align(x, rois, 2.0, 14, 14, 2, True)
        test_case.assertTrue(
            "The dimension of rois tensor must be equal to 2, but got"
            in str(ctx.exception)
        )

    def test_rol_align_rois_tensor_size_err(test_case):
        x = flow.randn(2, 3, 64, 5)
        rois = flow.randn(2, 3)
        with test_case.assertRaises(RuntimeError) as ctx:
            flow.roi_align(x, rois, 2.0, 14, 14, 2, True)
        test_case.assertTrue(
            "The size of rois tensor must be equal to 5 at dimension 1, but got"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
