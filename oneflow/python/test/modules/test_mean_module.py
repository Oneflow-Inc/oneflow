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
import oneflow as flow
import unittest
import numpy as np


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_sum(test_case):
        mean = flow.Mean(axis=1)
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = mean(input)
        np_out = np.mean(input.numpy(), axis=1)
        assert np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4)

        mean = flow.Mean(axis=0)
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = mean(input)
        np_out = np.mean(input.numpy(), axis=0)
        assert np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4)


if __name__ == "__main__":
    unittest.main()
