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

import oneflow
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestForwardGraph(flow.unittest.TestCase):
    def test_forward_graph(test_case):
        class SubModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.nn.Parameter(flow.Tensor(6, 6))
                self.relu = flow.nn.ReLU()

            def forward(self, x, y):
                x = oneflow._C.matmul(x, self.weight)
                x = self.relu(x)
                y = self.relu(y)
                return (x, y)

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = SubModule()
                self.register_buffer("dummy_buff", flow.Tensor(6, 8))

            def forward(self, x, y):
                (x, y) = self.layer(x, y)
                x = oneflow._C.flatten(x, 1)
                x = oneflow._C.matmul(x, self.dummy_buff)
                return (x, y)

        class CustomGraph(flow.nn.Graph):
            def __init__(self, module):
                super().__init__()
                self.m = module

            def build(self, x, y):
                out = self.m(x, y)
                return out

        m = CustomModule()
        m.to("cuda")
        g = CustomGraph(m)
        x = flow.Tensor(6, 6)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        x = x.to("cuda")
        y = flow.Tensor(10, 10)
        flow.nn.init.uniform_(y, a=-1.0, b=1.0)
        y = y.to("cuda")
        print(repr(g))
        (z, a) = g._compile(x, y)
        test_case.assertEqual(z.shape, (6, 8))
        test_case.assertEqual(z.is_lazy, False)
        test_case.assertEqual(a.shape, (10, 10))
        test_case.assertEqual(a.is_lazy, False)
        print("graph proto: ", g._graph_proto)

    def test_add_backward(test_case):
        linear = flow.nn.Linear(3, 8)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        class GraphAddBackward(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.mean()
                out.backward()
                return out

        g_with_b = GraphAddBackward()
        x = flow.ones(8, 3)
        out = g_with_b(x)
        print("graph proto: ", g_with_b._graph_proto)
        print(repr(g_with_b))


if __name__ == "__main__":
    unittest.main()
