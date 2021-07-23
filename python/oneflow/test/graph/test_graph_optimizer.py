import os
import unittest

import numpy as np

import oneflow
import oneflow.experimental as flow


@flow.unittest.skip_unless_1n1d()
class TestGraphOptimizer(flow.unittest.TestCase):
    def test_optimizer(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.para0 = flow.nn.Parameter(flow.Tensor(1, 4))
                self.para1 = flow.nn.Parameter(flow.Tensor(1, 4))
                self.para2 = flow.nn.Parameter(flow.Tensor(1, 4))
                self.para2.requires_grad_(False)
                self.para3 = flow.nn.Parameter(flow.Tensor(1, 4))
                self.para4 = flow.nn.Parameter(flow.Tensor(1, 4))

            def forward(self, x):
                return x

        m = CustomModule()
        learning_rate = 0.1
        momentum = 0.2
        scale = 0.3
        sgd0 = flow.optim.SGD(
            [
                {
                    "params": [m.para0, m.para1, m.para2],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "scale": scale,
                }
            ]
        )
        sgd1 = flow.optim.SGD(
            [
                {
                    "params": [m.para3],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "scale": scale,
                },
                {
                    "params": [m.para4],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "scale": scale,
                },
            ]
        )

        class CustomGraph0(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m
                self.add_optimizer("sgd0", sgd0)
                self.add_optimizer("sgd1", sgd1)

            def build(self, x):
                out = self.m(x)
                out.backward()
                return out

        g = CustomGraph0()
        x = flow.Tensor(1, 1, 10, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        z = g._compile(x)
        print(repr(g))
        print("g.config.proto: \n", g.config.proto)
        print("graph proto: \n", g._graph_proto)


if __name__ == "__main__":
    unittest.main()
