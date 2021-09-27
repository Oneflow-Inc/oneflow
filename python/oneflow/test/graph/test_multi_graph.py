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
import os
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestMultiGraph(oneflow.unittest.TestCase):
    def test_multi_graph(test_case):
        relu_data = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        relu_in = flow.tensor(relu_data, dtype=flow.float32)

        MyRelu = flow.nn.ReLU()
        relu_out_eager = MyRelu(relu_in)

        class ReluGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.cc_relu = MyRelu

            def build(self, x):
                return self.cc_relu(x)

        relu_g = ReluGraph()
        relu_out_lazy = relu_g(relu_in)
        test_case.assertTrue(
            np.array_equal(relu_out_lazy.numpy(), relu_out_eager.numpy())
        )

        linear = flow.nn.Linear(3, 8, False)
        linear = linear.to(flow.device("cuda"))
        input_arr = np.array(
            [
                [-0.94630778, -0.83378579, -0.87060891],
                [2.0289922, -0.28708987, -2.18369248],
                [0.35217619, -0.67095644, -1.58943879],
                [0.08086036, -1.81075924, 1.20752494],
                [0.8901075, -0.49976737, -1.07153746],
                [-0.44872912, -1.07275683, 0.06256855],
                [-0.22556897, 0.74798368, 0.90416439],
                [0.48339456, -2.32742195, -0.59321527],
            ],
            dtype=np.float32,
        )
        np_weight = np.ones((3, 8)).astype(np.float32)
        np_weight.fill(2.3)
        linear_in = flow.tensor(input_arr, device=flow.device("cuda"))
        flow.nn.init.constant_(linear.weight, 2.3)
        linear_out_eager = linear(linear_in)
        np_out = np.matmul(input_arr, np_weight)
        test_case.assertTrue(
            np.allclose(linear_out_eager.numpy(), np_out, 1e-05, 1e-05)
        )

        class LinearGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.my_linear = linear

            def build(self, x):
                return self.my_linear(x)

        linear_g = LinearGraph()
        linear_out_lazy = linear_g(linear_in)
        test_case.assertTrue(
            np.array_equal(linear_out_lazy.numpy(), linear_out_eager.numpy())
        )

        relu_out_lazy = relu_g(relu_in)
        linear_out_lazy = linear_g(linear_in)
        test_case.assertTrue(
            np.array_equal(relu_out_eager.numpy(), relu_out_lazy.numpy())
        )
        test_case.assertTrue(
            np.array_equal(linear_out_eager.numpy(), linear_out_lazy.numpy())
        )


if __name__ == "__main__":
    unittest.main()
