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
from itertools import product

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def get_dtype_str(dtype):
    return str(dtype).split(".")[-1]


dtype_list = [
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
]


@flow.unittest.skip_unless_1n1d()
class TestBinaryMathOpsDtype(flow.unittest.TestCase):
    @autotest(n=2, auto_backward=False, check_graph=False)
    def test_binary_math_ops_dtype(test_case):
        device = random_device()

        for x1_dtype, x2_dtype in product(dtype_list, dtype_list):
            x1 = random_tensor(2, 2, 3, requires_grad=False).to(device).to(x1_dtype)
            x2 = random_tensor(2, 2, 3, requires_grad=False).to(device).to(x2_dtype)

            for op in ["+", "-", "*", "/"]:
                y = eval(f"x1 {op} x2")
                test_case.assertEqual(
                    get_dtype_str(y.oneflow.dtype), get_dtype_str(y.pytorch.dtype)
                )


if __name__ == "__main__":
    unittest.main()
