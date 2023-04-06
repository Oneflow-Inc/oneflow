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


def _check_forward_and_backward(test_case, input, of_out, torch_out):
    # compare forward
    test_case.assertTrue(
        np.array_equal(of_out.numpy(), torch_out.cpu().detach().numpy())
    )

    # compare backward
    of_out.sum().backward()
    torch_out.sum().backward()
    torch_grad_local = input.pytorch.grad.cpu().detach()
    test_case.assertTrue(
        np.array_equal(input.oneflow.grad.numpy(), torch_grad_local.numpy())
    )


def _test_slice_random_data(test_case, placement, sbp):
    dims = [random(1, 2) * 8 for _ in range(2)]
    input = random_tensor(2, *dims)
    x = input.to_global(placement=placement, sbp=sbp)
    slice_tup_list = [[None, None, None], [0, 5, 2]]
    of_out = flow.slice(x.oneflow, slice_tup_list=slice_tup_list)
    torch_out = x.pytorch[:, 0:5:2]

    _check_forward_and_backward(test_case, input, of_out, torch_out)


def _test_slice_empty(test_case, placement, sbp):
    dims = [random(1, 2) * 8 for _ in range(2)]
    input = random_tensor(2, *dims)
    x = input.to_global(placement=placement, sbp=sbp)
    slice_tup_list = [[3, 3, 1], [None, None, None]]
    of_out = flow.slice(x.oneflow, slice_tup_list=slice_tup_list)
    torch_out = x.pytorch[3:3:1, :]

    _check_forward_and_backward(test_case, input, of_out, torch_out)


def _test_slice_1dim(test_case, placement, sbp):
    dims = [random(1, 2) * 8 for _ in range(2)]
    input = random_tensor(2, *dims)
    x = input.to_global(placement=placement, sbp=sbp)
    of_out = x.oneflow[2]
    torch_out = x.pytorch[2]

    _check_forward_and_backward(test_case, input, of_out, torch_out)


def _test_negative_index(test_case, placement, sbp):
    dims = [random(1, 2) * 8 for _ in range(2)]
    input = random_tensor(2, *dims)
    x = input.to_global(placement=placement, sbp=sbp)
    of_out = x.oneflow[-1:-6:1, :]
    torch_out = x.pytorch[-1:-6:1, :]

    _check_forward_and_backward(test_case, input, of_out, torch_out)


def _test_slice_ellipsis_type(test_case, placement, sbp):
    dims = [random(1, 2) * 8 for _ in range(2)]
    input = random_tensor(2, *dims)
    x = input.to_global(placement=placement, sbp=sbp)
    of_out = x.oneflow[..., :]
    torch_out = x.pytorch[..., :]

    _check_forward_and_backward(test_case, input, of_out, torch_out)


def _test_slice_with_bool(test_case, placement, sbp):
    x = random_tensor(2, 8, 8).oneflow > 0.5
    x_numpy = x.detach().cpu().numpy()

    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.slice(x, slice_tup_list=[[0, 1, 1]])

    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy[0:1:1]))


@autotest(
    n=2, auto_backward=False, check_graph=True,
)
def _test_slice_with_grad(test_case, placement):
    sbp = random_sbp(placement, max_dim=2).value()

    # out_sbp
    sbp_map = {
        flow.sbp.broadcast: flow.sbp.broadcast,
        flow.sbp.split(0): flow.sbp.split(0),
        flow.sbp.split(1): flow.sbp.partial_sum(),
        flow.sbp.partial_sum: flow.sbp.partial_sum(),
    }
    assert sbp is not None
    out_sbp = tuple([sbp_map[in_sbp] for in_sbp in sbp])

    x = random_tensor(2, 8, 16, requires_grad=True).oneflow
    x_numpy = x.detach().cpu().numpy()

    class SliceWithGrad(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_grad = flow.nn.Parameter(flow.zeros(8, 16))

        def forward(self, input):
            x = input + self.input_grad
            x = x.to_global(placement, sbp)
            return x[:, :8]

    slice_with_grad_m = SliceWithGrad().to_global(
        placement, [flow.sbp.broadcast,] * len(sbp)
    )

    of_sgd = flow.optim.SGD(slice_with_grad_m.parameters(), lr=1.0, momentum=0.0)

    class SliceTrainGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.module = slice_with_grad_m
            self.add_optimizer(of_sgd)

        def build(self, x):
            out = self.module(x)
            test_case.assertEqual(
                out.sbp,
                out_sbp,
                f"input sbp is {sbp}, but output sbp is {out.sbp} with placement: {placement}",
            )
            z = out.sum()
            z.backward()
            return out

    graph = SliceTrainGraph()

    input = x.to_global(placement=placement, sbp=sbp)
    y = graph(input)

    # output
    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy[:, :8]))
    # input_grad
    x_grad_np = np.zeros((8, 16))
    x_grad_np[:, :8] = 1
    test_case.assertTrue(
        np.array_equal(-graph.module.input_grad.to(flow.Tensor).numpy(), x_grad_np)
    )


class TestSlice(flow.unittest.TestCase):
    @globaltest
    def test_slice(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_slice_random_data(test_case, placement, sbp)
                _test_slice_empty(test_case, placement, sbp)
                _test_slice_1dim(test_case, placement, sbp)
                _test_negative_index(test_case, placement, sbp)
                _test_slice_ellipsis_type(test_case, placement, sbp)
                _test_slice_with_bool(test_case, placement, sbp)

    @globaltest
    def test_graph_slice(test_case):
        for placement in all_placement():
            _test_slice_with_grad(test_case, placement)


if __name__ == "__main__":
    unittest.main()
