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
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_logical_slice_assign(test_case, placement, sbp):
    input = random_tensor(2, 4, 4, requires_grad=True).oneflow
    x_numpy = input.detach().cpu().numpy()

    x = (input + 0).to_global(
        placement=placement, sbp=sbp
    )  # add 0 to change to non-leaf tensor
    x[:, :2] = 3

    # forward
    x_numpy[:, :2] = 3
    test_case.assertTrue(x.sbp == sbp)
    test_case.assertTrue(np.array_equal(x.numpy(), x_numpy))

    # backward
    x.sum().backward()
    input_grad_np = np.ones((4, 4))
    input_grad_np[:, :2] = 0
    test_case.assertTrue(np.array_equal(input.grad.numpy(), input_grad_np))


def _test_graph_logical_slice_assign(test_case, placement, sbp):
    x = random_tensor(2, 4, 4, requires_grad=True).oneflow
    x_numpy = x.detach().cpu().numpy()

    class LogicalSliceAssignWithGrad(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_grad = flow.nn.Parameter(flow.zeros(4, 4))

        def forward(self, input):
            x = input + self.input_grad
            x = x.to_global(placement, sbp)
            x[:, :2] = 3
            return x

    logical_slice_assign_with_grad = LogicalSliceAssignWithGrad().to_global(
        placement, [flow.sbp.broadcast,] * len(sbp)
    )

    of_sgd = flow.optim.SGD(
        logical_slice_assign_with_grad.parameters(), lr=1.0, momentum=0.0
    )

    class LogicalSliceAssignTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.module = logical_slice_assign_with_grad
            self.add_optimizer(of_sgd)

        def build(self, x):
            out = self.module(x)
            z = out.sum()
            z.backward()
            return out

    graph = LogicalSliceAssignTrainGraph()

    input = x.to_global(placement=placement, sbp=sbp)
    y = graph(input)

    test_case.assertTrue(y.sbp == sbp)

    # output
    x_numpy[:, :2] = 3
    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy))
    # input_grad
    x_grad_np = np.ones((4, 4))
    x_grad_np[:, :2] = 0
    test_case.assertTrue(
        np.array_equal(-graph.module.input_grad.origin.numpy(), x_grad_np)
    )


class TestGlobalLogicalSliceAssign(flow.unittest.TestCase):
    @globaltest
    def test_logical_slice_assign(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):
                if placement.ranks.size == 1:
                    continue
                _test_logical_slice_assign(test_case, placement, sbp)
                _test_graph_logical_slice_assign(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
