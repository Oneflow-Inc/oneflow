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
import os
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_linear_train_graph(test_case, device):
    def train_with_module(iter_num=3):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2)
        flow.nn.init.constant_(linear.bias, 0)

        x = flow.ones(8, 3).to(device)
        x.requires_grad = True

        for param in linear.parameters():
            param.requires_grad = False

        class LinearWithGrad(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.input_grad = flow.nn.Parameter(flow.zeros(8, 3))

            def forward(self, input):
                x = input + self.input_grad
                x = self.linear(x)
                return x

        linear_with_grad = LinearWithGrad().to(device)

        def one_iter():
            of_out = linear_with_grad(x)
            of_out = of_out.sum()
            of_out.backward()

            return {
                "out": of_out.numpy(),
                "weight": linear.weight.numpy(),
                "input grad": x.grad,
            }

        for i in range(iter_num):
            print("--- eager iter ", i)
            print(one_iter())

    def train_with_graph(iter_num=3):
        linear = flow.nn.Linear(3, 8)
        flow.nn.init.constant_(linear.weight, 2)
        flow.nn.init.constant_(linear.bias, 0)

        x = flow.ones(8, 3).to(device)

        for param in linear.parameters():
            param.requires_grad = False

        class LinearWithGrad(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.input_grad = flow.nn.Parameter(flow.zeros(8, 3))

            def forward(self, input):
                x = input + self.input_grad
                x = self.linear(x)
                return x

        linear_with_grad = LinearWithGrad().to(device)

        of_sgd = flow.optim.SGD(linear_with_grad.parameters(), lr=1.0, momentum=0.0)

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear_with_grad
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraph()
        print("graph: ", linear_t_g)
        print({"grad": linear_t_g.linear.input_grad.origin.numpy()})

        def one_iter():
            of_graph_out = linear_t_g(x)
            return {
                "out": of_graph_out.numpy(),
                "weight": linear_t_g.linear.linear.weight.origin.numpy(),
                "input grad": linear_t_g.linear.input_grad.origin.numpy(),
            }

        for i in range(iter_num):
            print("+++ graph iter ", i)
            print(one_iter())

    iter_num = 3
    train_with_module(iter_num)
    train_with_graph(iter_num)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearTrainGraph(oneflow.unittest.TestCase):
    def test_linear_train_graph_gpu(test_case):
        _test_linear_train_graph(test_case, flow.device("cuda"))


if __name__ == "__main__":
    unittest.main()
