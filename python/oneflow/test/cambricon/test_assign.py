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

import numpy as np
import oneflow.unittest
import oneflow as flow


def _test_assign(test_case, dtype):
    arr = np.random.randn(4, 5, 6, 7).astype(np.float32)
    input = flow.tensor(arr, device="mlu")
    x = input.permute(0, 3, 2, 1)  # x is non-contiguous tensor
    test_case.assertTrue(x.is_contiguous() == False)
    # y1 is normal version of tensor contiguous
    y1 = x.contiguous()
    # y2 is inplace version of tensor contiguous
    # assign kernel exec in inplace contiguous
    y2 = x.contiguous_()
    test_case.assertTrue(np.array_equal(y1.cpu().numpy(), y2.cpu().numpy()))
    test_case.assertTrue(id(x) != id(y1))
    test_case.assertTrue(id(x) == id(y2))
    test_case.assertTrue(x.is_contiguous() == True)
    test_case.assertTrue(y1.is_contiguous() == True)
    test_case.assertTrue(y2.is_contiguous() == True)


@flow.unittest.skip_unless_1n1d()
class TestAssign(flow.unittest.TestCase):
    def test_assign(test_case):
        dtype_list = [
            flow.float32,
            flow.float16,
            flow.int8,
            flow.uint8,
            flow.int32,
        ]
        for dtype in dtype_list:
            _test_assign(test_case, dtype)


if __name__ == "__main__":
    unittest.main()
