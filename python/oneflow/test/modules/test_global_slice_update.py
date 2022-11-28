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


def _test_slice_update(test_case, placement, sbp):
    input = random_tensor(2, 8, 16, requires_grad=True).oneflow
    value = random_tensor(2, 8, 8, requires_grad=True).oneflow
    x = (input + 0).to_global(
        placement=placement, sbp=sbp
    )  # add 0 to change to non-leaf tensor
    y = value.to_global(placement, sbp=sbp)
    x[:, :8] = y

    ref_np = input.detach().cpu().numpy()
    value_np = value.detach().cpu().numpy()

    # forward
    ref_np[:, :8] = value_np
    test_case.assertTrue(x.sbp == sbp)
    test_case.assertTrue(np.array_equal(x.numpy(), ref_np))

    # backward
    x.sum().backward()
    # ref grad
    ref_grad_np = np.ones((8, 16))
    ref_grad_np[:, :8] = 0
    test_case.assertTrue(np.array_equal(input.grad.numpy(), ref_grad_np))
    # value grad
    value_grad_np = np.ones((8, 8))
    test_case.assertTrue(np.array_equal(value.grad.numpy(), value_grad_np))


def _test_graph_slice_update(test_case, placement, sbp):
    ref = random_tensor(2, 8, 16, requires_grad=True).oneflow
    value = random_tensor(2, 8, 8, requires_grad=True).oneflow

    class SliceUpdateWithGrad(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.ref_grad = flow.nn.Parameter(flow.zeros(8, 16))
            self.value_grad = flow.nn.Parameter(flow.zeros(8, 8))

        def forward(self, ref, value):
            x = ref + self.ref_grad
            y = value + self.value_grad
            x = x.to_global(placement, sbp)
            y = y.to_global(placement, sbp)
            x[:, :8] = y
            return x

    slice_update_with_grad_m = SliceUpdateWithGrad().to_global(
        placement, [flow.sbp.broadcast,] * len(sbp)
    )

    of_sgd = flow.optim.SGD(slice_update_with_grad_m.parameters(), lr=1.0, momentum=0.0)

    class SliceUpdateTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.module = slice_update_with_grad_m
            self.add_optimizer(of_sgd)

        def build(self, x, y):
            out = self.module(x, y)
            z = out.sum()
            z.backward()
            return out

    graph = SliceUpdateTrainGraph()

    x = ref.to_global(placement=placement, sbp=sbp)
    y = value.to_global(placement=placement, sbp=sbp)
    z = graph(x, y)

    test_case.assertTrue(z.sbp == sbp)

    ref_np = ref.detach().cpu().numpy()
    value_np = value.detach().cpu().numpy()

    # forward
    ref_np[:, :8] = value_np
    test_case.assertTrue(np.array_equal(z.numpy(), ref_np))

    # backward
    # ref grad
    ref_grad = np.ones((8, 16))
    ref_grad[:, :8] = 0
    test_case.assertTrue(
        np.array_equal(-graph.module.ref_grad.to(flow.Tensor).numpy(), ref_grad)
    )
    # value grad
    value_grad = np.ones((8, 8))
    test_case.assertTrue(
        np.array_equal(-graph.module.value_grad.to(flow.Tensor).numpy(), value_grad)
    )


class TestGlobalSliceUpdate(flow.unittest.TestCase):
    @globaltest
    def test_slice_update(test_case):
        for placement in all_placement():
            for _ in range(2):
                sbp = random_sbp(placement, max_dim=2).value()
                _test_slice_update(test_case, placement, sbp)
                _test_graph_slice_update(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
