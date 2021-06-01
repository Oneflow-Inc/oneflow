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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPermute(flow.unittest.TestCase):
    def test_tensor_permute(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = input.permute(1, 0, 2, 3)
        np_out = input.numpy().transpose((1, 0, 2, 3))
        test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))

    def test_permute_negative_dim(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = flow.permute(input, -3, -4, 2, 3)
        np_out = input.numpy().transpose((1, 0, 2, 3))
        test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


if __name__ == "__main__":
    unittest.main()
