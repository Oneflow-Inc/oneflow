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


@flow.unittest.skip_unless_1n1d()
class TestTensorOperators(flow.unittest.TestCase):
    @autotest(n=5, check_graph=False)
    def test_tensor_inplace_operators_with_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x_dptr = x.oneflow.data_ptr()
        x_id = id(x.oneflow)
        x -= 1
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x /= 3
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x += 5
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        x *= 7
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        return x

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_tensor_inplace_operators_without_grad(test_case):
        device = random_device()
        x = random_tensor().to(device)
        x_dptr = x.oneflow.data_ptr()
        x_id = id(x.oneflow)
        x //= 2
        test_case.assertEqual(x_dptr, x.oneflow.data_ptr())
        test_case.assertEqual(x_id, id(x.oneflow))
        return x


if __name__ == "__main__":
    unittest.main()
