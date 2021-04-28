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
class TestTranspose(flow.unittest.TestCase):
    def test_transpose_v1(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = flow.tmp.transpose(input, perm=(0, 2, 3, 1))
        np_out = input.numpy().transpose((0, 2, 3, 1))
        test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))

    def test_transpose_by_tensor(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = input.transpose(perm=(0, 2, 3, 1))
        np_out = input.numpy().transpose((0, 2, 3, 1))
        test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


if __name__ == "__main__":
    unittest.main()
