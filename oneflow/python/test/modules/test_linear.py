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
import oneflow.typing as tp


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    

    def test_identity(test_case):
        m = flow.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        input_arr = np.array(
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
        x = flow.Tensor(input_arr)
        y = m(x)
        print(np.allclose(y.numpy(), y.numpy()))


if __name__ == "__main__":
    unittest.main()
