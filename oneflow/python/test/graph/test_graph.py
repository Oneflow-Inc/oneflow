import unittest

import numpy as np

import oneflow.experimental as flow
import oneflow

@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGraph(flow.unittest.TestCase):
    def test_add_nested_module(test_case):
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
            
            def forward(self, x):
                x = self.layer(x)
                x = oneflow.F.flatten(x, 1)
                x = self.fc1(x)
                return x

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
        test_case.assertTrue(isinstance(g.m, flow.experimental.nn.graph.Node))
        test_case.assertEqual(g.m.name, "m")
        test_case.assertTrue(isinstance(g.m.layer.conv1, flow.experimental.nn.graph.Node))
        test_case.assertEqual(g.m.layer.conv1.name, "conv1")
        test_case.assertEqual(g.m.layer.conv1.kernel_size, (5, 5))
        test_case.assertTrue(isinstance(g.m.layer.conv1.weight, flow.Tensor))
        test_case.assertTrue(isinstance(g.m.layer.conv1._parameters["weight"], flow.experimental.nn.graph.Node))

        # Graph build
        z = g.build(x)
        test_case.assertTrue(np.array_equal(y.numpy(), z.numpy()))
    
    # TODO(): test_graph_config
    # TODO(): test_add_optimizer


if __name__ == "__main__":
    unittest.main()