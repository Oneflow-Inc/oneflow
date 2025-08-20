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
from numpy import array, dtype
import oneflow as flow
import oneflow.unittest


class TestGeluOp(flow.unittest.TestCase):
    def TestGeluOp_inputshape_error_msg(test_case):
        a = flow.tensor([1, 2])
        b = flow.tensor([3, 4])
        c = flow.tensor([[2, 2], [2, 2]])
        with test_case.assertRaises(RuntimeError) as context:
            flow.add(a, b, c)
        test_case.assertTrue(
            "Expected multiplier shape same as the input tensor shape, but got " in str(context.exception) 
        )

    def TestGeluOp_inputtype_error_msg(test_case):
        with test_case.assertRaises(RuntimeError) as context:
            x = flow.tensor([1])
            indice = flow.tensor(1)
            flow.batch_gather(x, indice)
        test_case.assertTrue(
            "Expected multiplier data type same as the input tensor shape, but got " in str(context.exception)
        )

if __name__ == "__main__":
    unittest.main()