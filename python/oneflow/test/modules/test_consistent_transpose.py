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

from oneflow.test_utils.automated_test_util import *
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_consistent_transpose(test_case, placement, sbp):
    input = flow.tensor(
        np.random.randn(8, 16, 8, 16), dtype=flow.float32
    ).to_consistent(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    input = input.to_consistent(placement, sbp)
    of_out = flow.transpose(input, 0, 1)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_consistent_tensor_transpose(test_case, placement, sbp):
    input = flow.tensor(
        np.random.randn(8, 16, 8, 16), dtype=flow.float32
    ).to_consistent(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    input = input.to_consistent(placement, sbp)
    of_out = input.transpose(0, 1)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_consistent_tranpose_negative_dim(test_case, placement, sbp):
    input = flow.tensor(
        np.random.randn(8, 16, 8, 16), dtype=flow.float32
    ).to_consistent(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    input = input.to_consistent(placement, sbp)
    of_out = flow.transpose(input, -4, -3)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_consistent_transpose_backward(test_case, placement, sbp):
    x = flow.tensor(
        np.random.randn(8, 16, 8, 16), dtype=flow.float32, requires_grad=True,
    ).to_consistent(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    x = x.to_consistent(placement, sbp)
    y = flow.transpose(x, 0, 1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((8, 16, 8, 16)), 1e-05, 1e-05)
    )


def _test_consistent_transpose_backward_v2(test_case, placement, sbp):
    x = flow.tensor(
        np.random.randn(8, 16, 8, 16), dtype=flow.float32, requires_grad=True,
    ).to_consistent(flow.env.all_device_placement("cpu"), flow.sbp.broadcast)
    x = x.to_consistent(placement, sbp)
    y = flow.transpose(x, 3, 1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((8, 16, 8, 16)), 1e-05, 1e-05)
    )


@autotest(check_graph=False)
def _test_consistent_transpose_flow_with_random_data(test_case, placement, sbp):
    x = random_pytorch_tensor(4, 8, 16, 24, 8).to_consistent(placement, sbp)
    y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
    return y


@autotest(auto_backward=False, check_graph=False)
def _test_consistent_transpose_with_0_size_data(test_case, placement, sbp):
    device = random_device()
    x = random_pytorch_tensor(4, 8, 16, 0, 8).to_consistent(placement, sbp)
    y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
    return y


class TestConsistentTranspose(flow.unittest.TestCase):
    @consistent
    def test_consistent_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_consistent_transpose,
            _test_consistent_tensor_transpose,
            _test_consistent_tranpose_negative_dim,
            _test_consistent_transpose_backward,
            _test_consistent_transpose_backward_v2,
        ]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=4):
                    arg[0](test_case, placement, sbp)

    @consistent
    def test_consistent_transpose_flow_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_consistent_transpose_flow_with_random_data(
                    test_case, placement, sbp
                )

    # TODO(): Support 0 size tensor for eager consistent, which will
    # be solved in https://github.com/Oneflow-Inc/oneflow/pull/7326
    # @consistent
    # def test_consistent_transpose_with_0_size_data(
    #     test_case, valid_split_axis=[0, 1, 3]
    # ):
    #     for placement in all_placement():
    #         for sbp in all_sbp(placement, max_dim=4):
    #             _test_consistent_transpose_with_0_size_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
