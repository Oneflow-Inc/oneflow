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
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.tensor(
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
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        def one_iter():
            of_out = linear(x)
            of_out = of_out.sum()

            of_out.backward()
            of_sgd.step()
            of_sgd.zero_grad()

            return of_out.numpy(), linear.weight.numpy()

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    def train_with_graph(iter_num=3):
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 0.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

        x = flow.tensor(
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
            dtype=flow.float32,
            device=device,
            requires_grad=False,
        )

        class LinearTrainGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.add_optimizer(of_sgd)

            def build(self, x):
                out = self.linear(x)
                out = out.sum()
                out.backward()
                return out

        linear_t_g = LinearTrainGraph()

        def one_iter():
            of_graph_out = linear_t_g(x)
            print(linear_t_g.linear)
            return (
                of_graph_out.numpy(),
                linear_t_g.linear.weight.to(flow.Tensor).numpy(),
            )

        check_list = []
        for i in range(iter_num):
            check_list.append(one_iter())
        return check_list

    iter_num = 3
    module_check_list = train_with_module(iter_num)
    graph_check_list = train_with_graph(iter_num)
    for i in range(iter_num):
        # check equal on loss
        test_case.assertTrue(
            np.array_equal(module_check_list[i][0], graph_check_list[i][0])
        )
        # check equal on weight
        test_case.assertTrue(
            np.array_equal(module_check_list[i][1], graph_check_list[i][1])
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearTrainGraph(oneflow.unittest.TestCase):
    def test_linear_train_graph_gpu(test_case):
        _test_linear_train_graph(test_case, flow.device("cuda"))

    def test_linear_train_graph_cpu(test_case):
        _test_linear_train_graph(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
