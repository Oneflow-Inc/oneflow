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
from oneflow._oneflow_internal import Tensor, TensorTuple


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphCheck(flow.unittest.TestCase):
    def test_tensor_numpy_check(test_case):
        class CustomModuleNumpyCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.numpy()
                return x

        class CustomGraphNumpyCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleNumpyCheck()

            def build(self, x):
                return self.m(x)

        g = CustomGraphNumpyCheck()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        try:
            out = g(x)
        except:
            print(sys.exc_info())

    def test_non_tensor_types_of_module(test_case):
        class CustomModuleIOCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t, tp, i, s, n):
                return t, tp, i, s, n

        t0 = np.ones((10, 10))
        t0 = flow.tensor(t0, dtype=flow.float32)
        t1 = np.ones((10, 10))
        t1 = flow.tensor(t1, dtype=flow.float32)
        tp0 = TensorTuple()
        tp0.append(t0)
        tp0.append(t1)

        class CustomGraphIOCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleIOCheck()

            def build(self, t):
                rt, rtp, ri, rs, n = self.m(t, tp0, 1, "2", None)
                return t

        g = CustomGraphIOCheck()
        g.debug()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))


if __name__ == "__main__":
    unittest.main()
