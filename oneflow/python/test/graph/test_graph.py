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

        class CustomGraph(flow.experimental.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m

            def build(self, x):
                return self.m(x)
        
        # Graph init
        g = CustomGraph()
        # g.m is Node
        test_case.assertTrue(isinstance(g.m, flow.experimental.nn.graph.Node))
        # g.m.name is "m"
        test_case.assertEqual(g.m.name, "m")
        # g.m.dummy_buff is Tensor, Graph.build(...) need buffer to be Tensor
        test_case.assertTrue(isinstance(g.m.dummy_buff, flow.Tensor))
        # g.m._buffers["dummy_buff"] is Node
        test_case.assertTrue(isinstance(g.m._buffers["dummy_buff"], flow.experimental.nn.graph.Node))
        # conv1 is Node
        test_case.assertTrue(isinstance(g.m.layer.conv1, flow.experimental.nn.graph.Node))
        # conv1.name is "conv1"
        test_case.assertEqual(g.m.layer.conv1.name, "conv1")
        # conv1.weight is Tensor, Graph.build(...) need weight to be Tensor
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.Tensor))
        # conv1._parameters["weight"] is Node
        test_case.assertTrue(isinstance(g.m.layer.conv1._parameters["weight"], flow.experimental.nn.graph.Node))
        # conv1.kernel_size is original data in original module
        test_case.assertEqual(g.m.layer.conv1.kernel_size, (5, 5))

        # Graph build
        z = g.build(x)
        # g got the same result as m
        test_case.assertTrue(np.array_equal(y.numpy(), z.numpy()))
    
    def test_graph_config(test_case):
        class CustomGraph(flow.experimental.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_auto_mixed_precision(True)

            def build(self, x):
                x = self.m(x)
                return x
        
        g = CustomGraph()

        g.train(True)
        test_case.assertEqual(g.training, True)
        test_case.assertEqual(g.m.training, True)
        test_case.assertEqual(g.m.layer.conv1.training, True)
        g.config.enable_fuse_add_to_output(True)
        print(g.config.proto)

        g.train(False)
        test_case.assertEqual(g.training, False)
        test_case.assertEqual(g.training, False)
        test_case.assertEqual(g.m.training, False)
        test_case.assertEqual(g.m.layer.conv1.training, False)

        g.config.enable_fuse_add_to_output(False)
        print(g.config.proto)

        print(g.config.enable_fuse_add_to_output)
        print(repr(g))

    # TODO(): test_add_optimizer


if __name__ == "__main__":
    unittest.main()