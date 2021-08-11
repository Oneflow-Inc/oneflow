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
import os
import unittest
import sys

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.framework.tensor import Tensor, TensorTuple


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphCheck(flow.unittest.TestCase):
    def test_non_tensor_types_of_module(test_case):
        class CustomModuleIOCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t, tp, lt, n, i, s):
                return t, tp, lt, n, i, s

        class CustomGraphIOCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleIOCheck()

            def build(self, t, tp, lt, n):
                rt, rtp, rlt, n, ri, rs = self.m(t, tp, lt, n, 1, "2")
                return t, tp, lt, n

        g = CustomGraphIOCheck()
        g.debug()

        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)

        t0 = np.ones((10, 10))
        t0 = flow.tensor(t0, dtype=flow.float32)
        t1 = np.ones((10, 10))
        t1 = flow.tensor(t1, dtype=flow.float32)
        tp0 = TensorTuple()
        tp0.append(t0)
        tp0.append(t1)

        t2 = np.ones((10, 10))
        t2 = flow.tensor(t2, dtype=flow.float32)
        t3 = np.ones((10, 10))
        t3 = flow.tensor(t3, dtype=flow.float32)
        lt0 = list()
        lt0.append(t2)
        lt0.append(t3)

        ot, otp, olt, on = g(x, tp0, lt0, None)
        test_case.assertTrue(np.array_equal(x.numpy(), ot.numpy()))

        test_case.assertTrue(isinstance(otp, TensorTuple))
        test_case.assertTrue(isinstance(otp[0], Tensor))
        test_case.assertTrue(np.array_equal(otp[0].numpy(), tp0[0].numpy()))
        test_case.assertTrue(isinstance(otp[1], Tensor))
        test_case.assertTrue(np.array_equal(otp[1].numpy(), tp0[1].numpy()))

        test_case.assertTrue(isinstance(olt, list))
        test_case.assertTrue(isinstance(olt[0], Tensor))
        test_case.assertTrue(np.array_equal(olt[0].numpy(), lt0[0].numpy()))
        test_case.assertTrue(isinstance(olt[1], Tensor))
        test_case.assertTrue(np.array_equal(olt[1].numpy(), lt0[1].numpy()))

        test_case.assertTrue(on is None)


if __name__ == "__main__":
    unittest.main()
