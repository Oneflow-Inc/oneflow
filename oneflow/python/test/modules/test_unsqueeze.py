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
        np_arr = np.random.rand(2, 6, 9, 3)
        x = flow.Tensor(np_arr)
        y = flow.unsqueeze(x, dim=1)
        output = np.expand_dims(np_arr, axis=1)
        test_case.assertTrue(np.allclose(output, y.numpy(), rtol=1e-05))

    def test_unsqueeze_tensor_function(test_case):
        np_arr = np.random.rand(2, 3, 4)
        x = flow.Tensor(np_arr)
        y = x.unsqueeze(dim=2)
        output = np.expand_dims(np_arr, axis=2)
        test_case.assertTrue(np.allclose(output, y.numpy(), rtol=1e-05))

    def test_unsqueeze_different_dim(test_case):
        np_arr = np.random.rand(4, 5, 6, 7)
        x = flow.Tensor(np_arr)
        for axis in range(-5, 5):
            y = flow.unsqueeze(x, dim=axis)
            output = np.expand_dims(np_arr, axis=axis)
            test_case.assertTrue(np.allclose(output, y.numpy(), rtol=1e-05))


if __name__ == "__main__":
    unittest.main()
