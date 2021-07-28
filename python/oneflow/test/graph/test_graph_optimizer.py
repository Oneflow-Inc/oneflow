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

import oneflow
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestGraphOptimizer(flow.unittest.TestCase):
    def test_optimizer(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.para0 = flow.nn.Parameter(flow.Tensor(10, 4))

            def forward(self, x):
                x = flow.F.matmul(x, self.para0)
                return x

        m = CustomModule()
        learning_rate = 0.1
        momentum = 0.2
        scale = 0.3
        weight_decay = 0.7
        sgd0 = flow.optim.SGD(
            [
                {
                    "params": [m.para0],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "scale": scale,
                    "weight_decay": weight_decay,
                }
            ]
        )

        class CustomGraph0(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m
                self.add_optimizer("sgd0", sgd0)

            def build(self, x):
                out = self.m(x)
                out.backward()
                return out

        g = CustomGraph0()
        x = flow.Tensor(4, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        z = g._compile(x)
        print("repr(g): \n", repr(g))
        print("g.config.proto: \n", g.config.proto)
        print("graph proto: \n", g._graph_proto)

    def test_multi_optimizer_conf(test_case):
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
                x = flow.F.matmul(self.para0, x)
                y = flow.F.matmul(self.para3, x)
                return x, y

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
                    "weight_decay": 0.3,
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
                    "weight_decay": 0.4,
                },
                {
                    "params": [m.para4],
                    "lr": learning_rate,
                    "momentum": 0.9,
                    "scale": scale,
                    "weight_decay": 0.5,
                },
            ]
        )

        class CustomGraph0(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = m
                self.add_optimizer("sgd0", sgd0)
                self.add_optimizer("sgd1", sgd1)

            def build(self, x, y):
                out0, out1 = self.m(x, y)
                out0.backward()
                out1.backward()
                return out0, out1

        g = CustomGraph0()
        x = flow.Tensor(4, 10)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        g._generate_optimizer_and_variable_configs()
        print("repr(g): \n", repr(g))
        print("g.config.proto: \n", g.config.proto)


if __name__ == "__main__":
    unittest.main()
