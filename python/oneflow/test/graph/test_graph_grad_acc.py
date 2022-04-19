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


def _test_grad_acc_graph(test_case, device):
    def get_linear_sgd():
        linear = flow.nn.Linear(3, 8)
        linear = linear.to(device)
        flow.nn.init.constant_(linear.weight, 2.068758)
        flow.nn.init.constant_(linear.bias, 1.23)
        of_sgd = flow.optim.SGD(linear.parameters(), lr=0.01, momentum=0.9)
        return linear, of_sgd

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
        device=device,
        requires_grad=False,
    )

    free_one = flow.tensor([1.0], device=device, requires_grad=False)
    eager_linear, eager_sgd = get_linear_sgd()
    eager_out_list = []
    eager_weight_list = []
    for i in range(12):
        index = (i % 4) * 2
        input = x[index : (index + 2)]  # NOTE(chengcheng): unpack x by slice
        # print("i = ", i, " input = ", input)
        of_out = eager_linear(input)
        of_out += free_one  # Test free eager tensor
        one = flow.ones(of_out.shape, dtype=of_out.dtype, device=of_out.device)
        of_out += one
        of_out = flow.reshape(of_out, shape=[-1])
        of_out = of_out.sum()
        loss = of_out * 0.25  # NOTE(chengcheng): scale loss by grad acc
        loss.backward()
        if (i + 1) % 4 == 0:
            eager_sgd.step()
            eager_sgd.zero_grad()
            eager_weight_list.append(eager_linear.weight.numpy())
            # print("of_eager_weight in step: ", i,
            #      " weight = ", eager_linear.weight.numpy())

        # print("of_eager_out : ", of_out.numpy())
        eager_out_list.append(of_out.numpy())

    graph_linear, graph_sgd = get_linear_sgd()
    graph_out_list = []
    graph_weight_list = []

    class LinearTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.linear = graph_linear
            self.add_optimizer(graph_sgd)
            self.config.set_gradient_accumulation_steps(4)

        def build(self, x):
            out = self.linear(x)
            out += free_one  # Test free eager tensor
            one = flow.ones(out.shape, dtype=out.dtype, device=out.device)
            out += one
            out = flow.reshape(out, shape=[-1])
            # print("out.shape: ", out.shape)
            loss = out.sum()
            loss.backward()
            return out, loss

    linear_t_g = LinearTrainGraph()
    for i in range(3):
        # NOTE(chengcheng): Graph call 1 step for 1 mini-batch(4 micro-batch)
        non_scalar_out, of_out = linear_t_g(x)
        # print("of_lazy_out : ", of_out.numpy())

        graph_out_list.append(of_out.numpy())
        graph_weight_list.append(graph_linear.weight.numpy())
        # print("of_lazy_weight in step: ", i,
        #       " weight = ", graph_linear.weight.numpy())

    for i in range(3):
        test_case.assertTrue(np.allclose(eager_weight_list[i], graph_weight_list[i]))
        for j in range(4):
            test_case.assertTrue(
                eager_out_list[i * 4 + j].item() == graph_out_list[i][j]
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGradAccGraph(oneflow.unittest.TestCase):
    def test_grad_acc_graph_gpu(test_case):
        _test_grad_acc_graph(test_case, flow.device("cuda"))

    def test_grad_acc_graph_cpu(test_case):
        _test_grad_acc_graph(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
