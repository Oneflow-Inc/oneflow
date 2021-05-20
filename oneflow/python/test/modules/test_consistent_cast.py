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
class TestConsistentCastReLUModule(flow.unittest.TestCase):
    def test_relu(test_case):
        relu = flow.nn.ReLU()
        empty = flow.nn.Empty()
        arr = np.random.randn(8, 16, 12, 5)
        np_out = np.maximum(0, arr)

        empty.consistent_cast(
            (["S(0)"], ["S(0)"]),
            (
                [flow.placement("cpu", ["0:0"], None)],
                [flow.placement("cpu", ["0:0"], None)],
            ),
        )
        x = flow.Tensor(arr)
        y = empty(x)
        of_out = relu(y)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()
