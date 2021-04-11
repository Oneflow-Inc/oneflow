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

import torch


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_identity(test_case):
        torch_in = np.array(
            [[0.6898692, 0.2402668, 0.10445952], [0.7910769, 0.7279353, 0.42036182]],
            dtype=np.float32,
        )

        torch_out = np.array(
            [
                [2.0345955, 2.0345955, 2.0345955, 2.0345955],
                [2.939374, 2.939374, 2.939374, 2.939374],
            ],
            dtype=np.float32,
        )

        m = flow.nn.Linear(3, 4)
        x = flow.Tensor(torch_in)
        flow.nn.init.ones_(m.weight)
        flow.nn.init.ones_(m.bias)
        y = m(x)
        print(np.allclose(torch_out, y.numpy(), atol=1e-4))
        test_case.assertTrue(np.allclose(torch_out, y.numpy(), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
