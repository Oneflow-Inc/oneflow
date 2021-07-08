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
import oneflow


class SubModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = flow.nn.Conv2d(1, 1, 5)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class CustomModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SubModule()
        self.fc1 = flow.nn.Linear(36, 4)
        self.register_buffer(
            "dummy_buff", flow.Tensor(1, 4),
        )

    def forward(self, x):
        x = self.layer(x)
        x = oneflow.F.flatten(x, 1)
        x = self.fc1(x) + self.dummy_buff
        return x


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGraph(flow.unittest.TestCase):
    def test_add_nested_module(test_case):
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)

        # Module init and call
        m = CustomModule()
        y = m(x)

        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m

            def build(self, x):
                return self.m(x)

        # Graph init
        g = CustomGraph()
        # g.m is Block
        test_case.assertTrue(isinstance(g.m, flow.nn.graph.Block))
        # g.m.name is "m"
        test_case.assertEqual(g.m.name, "m")
        # g.m.dummy_buff is Tensor, Graph.build(...) need buffer to be Tensor
        test_case.assertTrue(isinstance(g.m.dummy_buff, flow.Tensor))
        # g.m._buffers["dummy_buff"] is Block
        test_case.assertTrue(
            isinstance(g.m._buffers["dummy_buff"], flow.nn.graph.Block)
        )
        # conv1 is Block
        test_case.assertTrue(isinstance(g.m.layer.conv1, flow.nn.graph.Block))
        # conv1.name is "conv1"
        test_case.assertEqual(g.m.layer.conv1.name, "conv1")
        # conv1.weight is Tensor, Graph.build(...) need weight to be Tensor
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.Tensor))
        # conv1._parameters["weight"] is Block
        test_case.assertTrue(
            isinstance(g.m.layer.conv1._parameters["weight"], flow.nn.graph.Block)
        )
        # conv1.kernel_size is original data in original module
        test_case.assertEqual(g.m.layer.conv1.kernel_size, (5, 5))

        # Graph build
        z = g.build(x)
        # g got the same result as m
        test_case.assertTrue(np.array_equal(y.numpy(), z.numpy()))

    def test_graph_config(test_case):
        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_auto_mixed_precision(True)

            def build(self, x):
                x = self.m(x)
                return x

        g = CustomGraph()

        # check default training is True
        test_case.assertEqual(g.config.training, False)

        # set graph config
        g.config.enable_fuse_add_to_output(True)
        g.config.enable_fuse_add_to_output(False)

        # check _named_state get the right tensor
        for n, t in g._named_state():
            test_case.assertEqual(id(eval("g." + n)), id(t))

        # print repr of nn.Graph
        print(repr(g))

    def test_graph_compile(test_case):
        class CustomGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_auto_mixed_precision(True)

            def build(self, x):
                x = self.m(x)
                return x

        g = CustomGraph()
        test_case.assertEqual(g.name, g._c_nn_graph.name)


if __name__ == "__main__":
    unittest.main()
