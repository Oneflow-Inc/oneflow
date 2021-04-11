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
import oneflow.typing as tp

import numpy as np
import unittest


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_embedding(test_case):
        torch_weight = np.array(
            [
                [0.68258786, 0.6957856, 1.1829041],
                [1.0154, -1.0616943, 0.50303376],
                [0.29679507, 0.65562993, 1.0424724],
                [-0.42980736, -0.35347632, -0.15600166],
                [0.6763601, -0.24286619, -2.0873115],
                [-0.13371214, -0.5589277, 1.9173933],
                [0.08762296, 1.0264007, -0.67938024],
                [0.32019204, -0.26137325, -1.3534237],
                [-1.1555519, -0.67776406, 0.27372134],
                [1.0615997, -0.59715784, 1.9855849],
            ],
            dtype=np.float32,
        )

        torch_out = np.array(
            [
                [
                    [1.0154, -1.0616943, 0.50303376],
                    [0.29679507, 0.65562993, 1.0424724],
                    [0.6763601, -0.24286619, -2.0873115],
                    [-0.13371214, -0.5589277, 1.9173933],
                ],
                [
                    [0.6763601, -0.24286619, -2.0873115],
                    [-0.42980736, -0.35347632, -0.15600166],
                    [0.29679507, 0.65562993, 1.0424724],
                    [1.0615997, -0.59715784, 1.9855849],
                ],
            ],
            dtype=np.float32,
        )

        indices = flow.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int32)
        # m2 = flow.nn.Embedding(10, 3, _weight=flow.Tensor(torch_weight))
        m2 = flow.nn.Embedding(10, 3)
        y = m2(indices)
        print(np.allclose(y.numpy(), torch_out))


if __name__ == "__main__":
    unittest.main()
