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
class TestGraphBlock(flow.unittest.TestCase):
    def test_module_has_custom_func(test_case):
        class CustomModuleHasFunc(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.data_mem = 10

            def forward(self, x):
                return self._custom_func(x)

            def _custom_func(self, x):
                test_case.assertEqual(self.data_mem, 10)
                return x

        class CustomGraphHasFunc(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleHasFunc()

            def build(self, x):
                return self.m(x)

        g = CustomGraphHasFunc()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))

    def test_block_with_parameter(test_case):
        device = "cuda"
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.Tensor(
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
            device=device,
            requires_grad=False,
        )

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return self._forward_impl(x)

            def _forward_impl(self, x):
                test_case.assertTrue(isinstance(self.linear, flow.nn.graph.Block))
                return self.linear(x)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.m(x)
                out = out.sum()
                out.backward()
                test_case.assertTrue(self.m.linear.weight.is_lazy)
                return out

        linear_t_g = LinearTrainGraph()

        linear_t_g(x)


if __name__ == "__main__":
    unittest.main()
