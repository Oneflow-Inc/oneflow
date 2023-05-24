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
from collections import OrderedDict

import oneflow as flow
from oneflow.test_utils.test_util import GenArgDict
from oneflow.nn.utils.parameters_grouping import ContiguousParamsGroup as CPG
from oneflow.nn.parameter import Parameter
import oneflow.unittest


def np_allclose_with_shape(a, b, *args, **kwargs):
    return a.shape == b.shape and np.allclose(a, b, *args, **kwargs)


def module_grouping(test_case, device):
    class Model(flow.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            dtypes = [flow.float32, flow.float64]
            for i in range(10):
                self.register_parameter(
                    f"w{i}",
                    flow.nn.Parameter(
                        flow.tensor([i % 2 + 1, i % 2 + 1], dtype=dtypes[i % 2])
                    ),
                )

    m = Model().to(device)
    m.make_contiguous_params_group()
    cpg = CPG(
        list(m.parameters())
        + [flow.tensor([3, 3], dtype=flow.float32, requires_grad=True)]
    )

    test_case.assertTrue(len(m.cpg.grouped_parameters) == 2)
    test_case.assertTrue(len(m.cpg.grouped_grads) == 2)
    test_case.assertTrue(flow.max(m.cpg.grouped_parameters[0]) == 1)
    test_case.assertTrue(flow.max(m.cpg.grouped_parameters[1]) == 2)

    test_case.assertTrue(len(cpg.grouped_parameters) == 3)
    test_case.assertTrue(len(cpg.grouped_grads) == 3)
    test_case.assertTrue(flow.max(cpg.grouped_parameters[0]) == 1)
    test_case.assertTrue(flow.max(cpg.grouped_parameters[1]) == 2)
    test_case.assertTrue(flow.max(cpg.grouped_parameters[2]) == 3)


def direct_grouping(test_case, device):
    x = [
        Parameter(
            flow.tensor(
                [1, 2],
                device=flow.device(device),
                dtype=flow.float32,
                requires_grad=True,
            )
        ),
        Parameter(
            flow.tensor(
                [3, 4],
                device=flow.device(device),
                dtype=flow.float32,
                requires_grad=True,
            )
        ),
    ]
    cpg = CPG([[x[0]], [x[1]]])
    test_case.assertTrue(len(cpg.grouped_parameters) == 2)
    test_case.assertTrue(len(cpg.grouped_grads) == 2)


def global_grouping(test_case, device):
    x = flow.nn.Parameter(
        flow.zeros((10,), dtype=flow.float32, requires_grad=True).to_global(
            sbp=flow.sbp.broadcast, placement=flow.placement(device, [0])
        )
    )
    y = flow.nn.Parameter(
        flow.zeros((10,), dtype=flow.float32, requires_grad=True).to_global(
            sbp=flow.sbp.split(0), placement=flow.placement(device, [0])
        )
    )
    cpg = CPG([x, y], group_on_current_buffer=False)
    test_case.assertTrue(len(cpg.grouped_parameters) == 2)
    test_case.assertTrue(len(cpg.grouped_grads) == 2)


def multi_module_grad(test_case, device):
    class Module1(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = flow.nn.Parameter(flow.Tensor([1, 1]))
            self.w2 = flow.nn.Parameter(flow.Tensor([1, 1]))

        def forward(self, x):
            return x * self.w1 * self.w2

    class Module2(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = flow.nn.Parameter(flow.Tensor([2, 2]))
            self.w2 = flow.nn.Parameter(flow.Tensor([2, 2]))

        def forward(self, x):
            return x * self.w1 * self.w2

    m1 = Module1().to(device)
    m1.make_contiguous_params_group()
    m2 = Module2().to(device)
    m2.make_contiguous_params_group()
    optim1 = flow.optim.SGD(m1.parameters(), lr=1e-2, contiguous_params=True)
    optim2 = flow.optim.SGD(m2.parameters(), lr=1e-2, contiguous_params=True)
    x1 = flow.ones([1, 1]).to(device)
    x2 = flow.ones([2, 2]).to(device)
    flow.sum(m1(x1)).backward()
    flow.sum(m2(x2)).backward()

    for p in m1.parameters():
        test_case.assertTrue(
            np_allclose_with_shape(p.grad.numpy(), np.array([1.0, 1.0]))
        )

    for p in m2.parameters():
        test_case.assertTrue(
            np_allclose_with_shape(p.grad.numpy(), np.array([4.0, 4.0]))
        )


def multi_module_lifecycle(test_case, device):
    class Module1(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = flow.nn.Parameter(flow.Tensor([1, 1]))
            self.w2 = flow.nn.Parameter(flow.Tensor([1, 1]))

        def forward(self, x):
            return x * self.w1 * self.w2

    class Module2(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = flow.nn.Parameter(flow.Tensor([2, 2]))
            self.w2 = flow.nn.Parameter(flow.Tensor([2, 2]))

        def forward(self, x):
            return x * self.w1 * self.w2

    m1 = Module1().to(device)
    m1.make_contiguous_params_group()
    m2 = Module2().to(device)
    m2.make_contiguous_params_group()
    del m1
    cpg = CPG(list(m2.parameters()))
    test_case.assertTrue(len(cpg.grouped_parameters) == 1)


@flow.unittest.skip_unless_1n1d()
class TestCPG(flow.unittest.TestCase):
    def test_cpg(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgDict(arg_dict):
            device = arg["device"]
            module_grouping(test_case, device)
            direct_grouping(test_case, device)
            global_grouping(test_case, device)
            multi_module_lifecycle(test_case, device)
            multi_module_grad(test_case, device)


if __name__ == "__main__":
    unittest.main()
