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
    ".numpy() doesn't work in lazy mode",
)
class TestFlattenModule(flow.unittest.TestCase):
    def test_flatten(test_case):
        m = flow.nn.Flatten(start_dim=1)
        x = flow.Tensor(32, 2, 5, 5)
        flow.nn.init.uniform_(x)
        y = m(x)
        test_case.assertTrue(y.shape == flow.Size((32, 50)))
        test_case.assertTrue(np.array_equal(y.numpy().flatten(), x.numpy().flatten()))

        y2 = flow.tmp.flatten(x, start_dim=2)
        test_case.assertTrue(y2.shape == flow.Size((32, 2, 25)))
        test_case.assertTrue(np.array_equal(y2.numpy().flatten(), x.numpy().flatten()))

        y3 = x.flatten(start_dim=1)
        test_case.assertTrue(y3.shape == flow.Size((32, 50)))
        test_case.assertTrue(np.array_equal(y3.numpy().flatten(), x.numpy().flatten()))

        y4 = x.flatten(start_dim=1, end_dim=2)
        test_case.assertTrue(y4.shape == flow.Size((32, 10, 5)))
        test_case.assertTrue(np.array_equal(y4.numpy().flatten(), x.numpy().flatten()))


if __name__ == "__main__":
    unittest.main()
