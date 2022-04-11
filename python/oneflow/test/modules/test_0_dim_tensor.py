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
from collections import OrderedDict

import numpy as np

import oneflow as flow


from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def _test_0_dim_tensor(test_case, device):
    scalar = 9.999
    input_np = np.array(scalar)
    input = flow.tensor(input_np, device=device)

    test_case.assertEqual(input.numel(), 1)
    test_case.assertEqual(input.ndimension(), 0)

    x1 = flow.tensor(np.array(2), dtype=flow.float32, device=device)
    x2 = flow.tensor(np.array(3), dtype=flow.float32, device=device)
    y1 = x1 * x2
    y2 = x1 + x2
    test_case.assertEqual(y1.numpy(), 6.0)
    test_case.assertEqual(y2.numpy(), 5.0)


def _test_scalar_mul(test_case, device):
    for dim in range(5):
        test_case.assertEqual(
            np.ones([2] * dim).sum(), flow.ones([2] * dim, device=device).sum().numpy()
        )


def _test_slice(test_case, device):
    x = flow.tensor(np.arange(10), device=device)
    for i in range(x.numel()):
        scalar_i = x[i]
        test_case.assertEqual(i, scalar_i.numpy())
        test_case.assertEqual(scalar_i.numel(), 1)
        test_case.assertEqual(scalar_i.ndimension(), 0)


def _test_slice_backward(test_case, device):
    np_grad = np.zeros(10)
    x = flow.tensor(np.arange(10).astype(np.float32), device=device, requires_grad=True)
    for i in range(x.numel()):
        y = x[i]
        z = y.sum()
        z.backward()
        np_grad[i] = 1
        test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-04, 1e-04))

    x2 = flow.tensor(
        np.arange(100).astype(np.float32), device=device, requires_grad=True
    )
    y2 = x2[1:100]
    z2 = y2.sum()
    z2.backward()
    np_grad2 = np.ones(100)
    np_grad2[0] = 0
    test_case.assertTrue(np.allclose(x2.grad.numpy(), np_grad2, 1e-04, 1e-04))


def _test_slice_scalar_graph(test_case, device):
    x = flow.tensor(3.0, device=device)

    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = flow.nn.Parameter(
                flow.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            )

        def forward(self, x):
            return x * self.weight[3]

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


def _test_slice_scalar_train_graph(test_case, device):
    class MyModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = flow.nn.Parameter(
                flow.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            )

        def forward(self, x):
            return x * self.weight[3] + 1.0

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


@flow.unittest.skip_unless_1n1d()
class TestZeroDimensionTensor(flow.unittest.TestCase):
    def test_0_dim_tensor(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_0_dim_tensor,
            _test_scalar_mul,
            _test_slice,
            _test_slice_backward,
            _test_slice_scalar_graph,
            _test_slice_scalar_train_graph,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
