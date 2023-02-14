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


def cpg_grouping(test_case, device):
    init_seq = []

    for i in range(7):
        init_seq.append(np.random.uniform(size=(i + 1,)).astype(np.float32))

    def _test_cpg_grad_helper():
        a = Parameter(flow.Tensor(init_seq[0], device=flow.device(device)))
        b = Parameter(flow.Tensor(init_seq[1], device=flow.device(device)))
        c = Parameter(flow.Tensor(init_seq[2], device=flow.device(device)))
        d = Parameter(flow.Tensor(init_seq[3], device=flow.device(device)))
        e = Parameter(flow.Tensor(init_seq[4], device=flow.device(device)))
        g = Parameter(flow.Tensor(init_seq[6], device=flow.device(device)))

        loss = a.sum() + b.sum() + c.sum() + d.sum() + e.sum() + g.sum()
        loss.backward()
        return [
            a.grad.numpy(),
            b.grad.numpy(),
            c.grad.numpy(),
            d.grad.numpy(),
            e.grad.numpy(),
            g.grad.numpy(),
        ]

    def _test_cpg_regrouping():
        # groups0 = [[a, b], [c, d, e, f]]
        # groups1 = [[e, d, g], [a], [b, c]]
        # -> [[d, e], [g], [a], [b], [c]]

        a = Parameter(flow.Tensor(init_seq[0], device=flow.device(device)))
        b = Parameter(flow.Tensor(init_seq[1], device=flow.device(device)))
        c = Parameter(flow.Tensor(init_seq[2], device=flow.device(device)))
        d = Parameter(flow.Tensor(init_seq[3], device=flow.device(device)))
        e = Parameter(flow.Tensor(init_seq[4], device=flow.device(device)))
        f = Parameter(flow.Tensor(init_seq[5], device=flow.device(device)))
        g = Parameter(flow.Tensor(init_seq[6], device=flow.device(device)))

        groups0 = [[a, b], [c, d, e, f]]
        cpg = CPG(groups0)

        test_case.assertTrue(len(cpg.grouped_parameters) == 2)
        test_case.assertTrue(len(set([p._ref_tensor for p in [a, b, c, d, e, f]])) == 2)
        test_case.assertTrue(len(set([p._ref_tensor for p in [a, b]])) == 1)
        test_case.assertTrue(len(set([p._ref_tensor for p in [c, d, e, f]])) == 1)

        groups1 = [[e, d, g], [a], [b, c]]
        cpg = CPG(groups1)

        test_case.assertTrue(len(cpg.grouped_parameters) == 5)
        test_case.assertTrue(len(set([p._ref_tensor for p in [a, b, c, d, e, g]])) == 3)
        test_case.assertTrue(len(set([p._ref_tensor for p in [d, e]])) == 1)

        loss = a.sum() + b.sum() + c.sum() + d.sum() + e.sum() + g.sum()
        loss.backward()

        return [
            a.grad.numpy(),
            b.grad.numpy(),
            c.grad.numpy(),
            d.grad.numpy(),
            e.grad.numpy(),
            g.grad.numpy(),
        ]

    grouped_grad = _test_cpg_regrouping()
    grad = _test_cpg_grad_helper()

    for (x, y) in zip(grouped_grad, grad):
        test_case.assertTrue(np_allclose_with_shape(x, y))


@flow.unittest.skip_unless_1n1d()
class TestCPG(flow.unittest.TestCase):
    def test_cpg(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgDict(arg_dict):
            cpg_grouping(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
