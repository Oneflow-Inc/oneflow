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
from collections import OrderedDict

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest


class TestBinaryFunctorError(flow.unittest.TestCase):
    def test_add_inplace_runtime_error(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            y = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            x.add_(y)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

    def test_add_broad_cast_runtime_error(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((2, 3))
            y = flow.ones((2, 4))
            x.add_(y)
        test_case.assertTrue(
            "Tensor with shape (2,3) doesn't match the broadcast shape in an inplace operation"
            in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((3, 3))
            y = flow.ones((2, 3, 3))
            x.add_(y)
        test_case.assertTrue(
            "Can not expand origin shape (2,3,3) to (3,3)" in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            y = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            x.mul_(y)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((2, 3))
            y = flow.ones((2, 4))
            x.mul_(y)
        test_case.assertTrue(
            "Tensor with shape (2,3) doesn't match the broadcast shape in an inplace operation"
            in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((3, 3))
            y = flow.ones((2, 3, 3))
            x.mul_(y)
        test_case.assertTrue(
            "Can not expand origin shape (2,3,3) to (3,3)" in str(context.exception)
        )

    def test_div_inplace_runtime_error(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            y = flow.ones((4, 4), dtype=flow.float32, requires_grad=True)
            x.div_(y)
        test_case.assertTrue(
            "a leaf Tensor that requires grad is being used in an in-place operation"
            in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((2, 3))
            y = flow.ones((2, 4))
            x.div_(y)
        test_case.assertTrue(
            "Tensor with shape (2,3) doesn't match the broadcast shape in an inplace operation"
            in str(context.exception)
        )

        with test_case.assertRaises(RuntimeError) as context:
            x = flow.ones((3, 3))
            y = flow.ones((2, 3, 3))
            x.div_(y)
        test_case.assertTrue(
            "Can not expand origin shape (2,3,3) to (3,3)" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
