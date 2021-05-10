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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_unsqueeze(test_case):
        m = flow.nn.Unsqueeze()
        x = flow.Tensor(np.random.rand(16, 20))
        y = m(x, 1)
        test_case.assertTrue(np.allclose(flow.Size([16, 1, 20]), y.shape))

    def test_unsqueeze2(test_case):
        x2 = flow.Tensor(np.random.rand(2, 3, 4))
        y2 = x2.unsqueeze(2)
        test_case.assertTrue(np.allclose(flow.Size([2, 3, 1, 4]), y2.shape))

    def test_unsqueeze3(test_case):
        x3 = flow.Tensor(np.random.rand(2, 6, 9, 3))
        y3 = x3.unsqueeze(4)
        test_case.assertTrue(np.allclose(flow.Size([2, 6, 9, 3, 1]), y3.shape))

    def test_unsqueeze4(test_case):
        x3 = flow.Tensor(np.random.rand(2, 6, 9, 3))
        y3 = flow.unsqueeze(x3, 4)
        test_case.assertTrue(np.allclose(flow.Size([2, 6, 9, 3, 1]), y3.shape))


if __name__ == "__main__":
    unittest.main()
