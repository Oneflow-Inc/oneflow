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


def _test_scalar_graph(test_case, device):
    x = flow.tensor(3.0, device=device)

    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = flow.nn.Parameter(flow.tensor(5.0, device=device))

        def forward(self, x):
            return x * self.weight + 1.0

    my_module = MyModule()
    of_eager_out = my_module(x)

    class ScalarGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = my_module

        def build(self, x):
            return self.m(x)

    scalar_g = ScalarGraph()
    of_lazy_out = scalar_g(x)
    test_case.assertTrue(np.array_equal(of_lazy_out.numpy(), of_eager_out.numpy()))


def _test_scalar_train_graph(test_case, device):
    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = flow.nn.Parameter(flow.tensor(5.0, device=device))

        def forward(self, x):
            return x * self.weight + 1.0

    my_module = MyModule()
    of_sgd = flow.optim.SGD(my_module.parameters(), lr=0.001, momentum=0.9)
    eager_out_list = []
    for i in range(3):
        x = flow.tensor(i * 1.0, device=device, requires_grad=False)
        of_eager_out = my_module(x)
        of_eager_out.backward()
        of_sgd.step()
        of_sgd.zero_grad()
        eager_out_list.append(of_eager_out)

    lazy_module = MyModule()

    class ScalarTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = lazy_module
            of_sgd = flow.optim.SGD(lazy_module.parameters(), lr=0.001, momentum=0.9)
            # self.m = MyModule()
            # of_sgd = flow.optim.SGD(self.m.parameters(), lr=0.001, momentum=0.9)
            self.add_optimizer(of_sgd)

        def build(self, x):
            loss = self.m(x)
            loss.backward()
            return loss

    lazy_out_list = []
    scalar_g = ScalarTrainGraph()
    for i in range(3):
        x = flow.tensor(i * 1.0, device=device)
        of_lazy_out = scalar_g(x)
        lazy_out_list.append(of_lazy_out)

    for i in range(3):
        test_case.assertTrue(
            np.array_equal(lazy_out_list[i].numpy(), eager_out_list[i].numpy())
        )


def _test_scalar_global_train_graph(test_case, placement):
    sbp_b = flow.sbp.broadcast

    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = flow.nn.Parameter(flow.tensor(5.0))

        def forward(self, x):
            return x * self.weight + 1.0

    my_module = MyModule()

    of_sgd = flow.optim.SGD(my_module.parameters(), lr=0.001, momentum=0.9)
    eager_out_list = []
    for i in range(3):
        x = flow.tensor(i * 1.0, requires_grad=False)
        of_eager_out = my_module(x)
        of_eager_out.backward()
        of_sgd.step()
        of_sgd.zero_grad()
        eager_out_list.append(of_eager_out)

    lazy_module = MyModule()
    lazy_module.to_global(placement=placement, sbp=sbp_b)

    class ScalarTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = lazy_module
            of_sgd = flow.optim.SGD(lazy_module.parameters(), lr=0.001, momentum=0.9)
            self.add_optimizer(of_sgd)

        def build(self, x):
            loss = self.m(x)
            loss.backward()
            return loss

    lazy_out_list = []
    scalar_g = ScalarTrainGraph()
    for i in range(3):
        x = flow.tensor(i * 1.0, requires_grad=False)
        x = x.to_global(placement=placement, sbp=sbp_b)
        of_lazy_out = scalar_g(x)
        lazy_out_list.append(of_lazy_out)
    for i in range(3):
        test_case.assertTrue(
            np.array_equal(
                lazy_out_list[i].to_local().numpy(), eager_out_list[i].numpy()
            )
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestScalarGraph(oneflow.unittest.TestCase):
    def test_scalar_graph_gpu(test_case):
        _test_scalar_graph(test_case, flow.device("cuda"))

    def test_scalar_graph_cpu(test_case):
        _test_scalar_graph(test_case, flow.device("cpu"))

    def test_scalar_train_graph_gpu(test_case):
        _test_scalar_train_graph(test_case, flow.device("cuda"))

    def test_scalar_train_graph_cpu(test_case):
        _test_scalar_train_graph(test_case, flow.device("cpu"))

    def test_scalar_global_train_graph_gpu(test_case):
        _test_scalar_global_train_graph(test_case, flow.placement("cuda", ranks=[0]))

    def test_scalar_global_train_graph_cpu(test_case):
        _test_scalar_global_train_graph(test_case, flow.placement("cpu", ranks=[0]))


if __name__ == "__main__":
    unittest.main()
