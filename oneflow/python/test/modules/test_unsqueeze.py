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
class TestUnsqueeze(flow.unittest.TestCase):
    def test_unsqueeze(test_case):
        x = flow.Tensor(np.random.rand(2, 6, 9, 3))
        y = flow.unsqueeze(x, dim=1)
        test_case.assertTrue(np.allclose(flow.Size([2, 1, 6, 9, 3]), y.shape))

    def test_unsqueeze_tensor_function(test_case):
        x = flow.Tensor(np.random.rand(2, 3, 4))
        y = x.unsqueeze(dim=2)
        test_case.assertTrue(np.allclose(flow.Size([2, 3, 1, 4]), y.shape))


if __name__ == "__main__":
    unittest.main()
