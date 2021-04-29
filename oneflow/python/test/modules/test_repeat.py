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


def np_repeat(x, sizes):
    return np.tile(x, sizes)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_repeat_new_dim(test_case):
        input = flow.Tensor(np.random.randn(2, 4, 1, 3), dtype=flow.float32)
        sizes = (4, 3, 2, 3, 3)
        np_out = np_repeat(input.numpy(), sizes)
        of_out = input.repeat(sizes=sizes)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    def test_repeat_same_dim(test_case):
        input = flow.Tensor(np.random.randn(1, 2, 5, 3), dtype=flow.float32)
        sizes = (4, 2, 3, 19)
        of_out = input.repeat(sizes=sizes)
        np_out = np_repeat(input.numpy(), sizes)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
