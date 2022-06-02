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


def _test_logical_slice(test_case, placement, sbp):
    x = random_tensor(2, 8, 8, requires_grad=False).oneflow
    x_numpy = x.detach().cpu().numpy()

    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.logical_slice(x, slice_tup_list=[[0, 1, 1]])

    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy[0:1:1]))


def _test_logical_slice_with_bool(test_case, placement, sbp):
    x = random_tensor(2, 8, 8).oneflow > 0.5
    x_numpy = x.detach().cpu().numpy()

    x = x.to_global(placement=placement, sbp=sbp)
    y = flow.logical_slice(x, slice_tup_list=[[0, 1, 1]])

    test_case.assertTrue(np.array_equal(y.numpy(), x_numpy[0:1:1]))


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


class TestLogicalSlice(flow.unittest.TestCase):
    @globaltest
    def test_logical_slice(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_logical_slice(test_case, placement, sbp)
                _test_logical_slice_with_bool(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
