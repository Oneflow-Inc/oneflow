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
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList


def _test_tensor_scatter_nd_update(test_case, device):
    origin = flow.tensor(np.arange(8), dtype=flow.float, device=flow.device(device))
    indices = flow.tensor(
        np.array([[1], [6], [4]]), dtype=flow.int, device=flow.device(device)
    )
    update = flow.tensor(
        np.array([10.2, 5.1, 12.7]), dtype=flow.float, device=flow.device(device)
    )
    np_out = np.array([0.0, 10.2, 2.0, 3.0, 12.7, 5.0, 5.1, 7.0])
    output = flow.tensor_scatter_nd_update(origin, indices, update)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_tensor_scatter_nd_update_with_non_contiguous_input(test_case, device):
    # non-contiguous tensor with shape (2, 3, 4)
    origin = flow.tensor(
        np.ones((4, 3, 2)), dtype=flow.float, device=flow.device(device)
    ).permute(2, 1, 0)
    # indices with shape (3, 2)
    indices = flow.tensor(
        np.array([[0, 0], [1, 0], [1, 1]]), dtype=flow.int, device=flow.device(device)
    )
    # non-contiguous update with shape (3, 4)
    update = flow.tensor(
        np.zeros((4, 3)), dtype=flow.float, device=flow.device(device)
    ).T
    output = flow.tensor_scatter_nd_update(origin, indices, update)

    np_res = np.ones((2, 3, 4))
    np_res[0, 0] = 0
    np_res[1, 0] = 0
    np_res[1, 1] = 0
    test_case.assertTrue(np.array_equal(output.numpy(), np_res))


def _test_tensor_scatter_nd_update_t(test_case, device):
    origin = flow.tensor(
        np.arange(15).reshape(5, 3), dtype=flow.float, device=flow.device(device)
    )
    indices = flow.tensor(
        np.array([[0], [4], [2]]), dtype=flow.int, device=flow.device(device)
    )
    update = flow.tensor(
        np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        dtype=flow.float,
        device=flow.device(device),
    )
    np_out = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [3.0, 3.0, 3.0],
            [9.0, 10.0, 11.0],
            [2.0, 2.0, 2.0],
        ]
    )
    output = flow.tensor_scatter_nd_update(origin, indices, update)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_tensor_scatter_nd_update_backward(test_case, device):
    origin = flow.tensor(
        np.arange(8), dtype=flow.float, device=flow.device(device), requires_grad=True,
    )
    indices = flow.tensor(
        np.array([[1], [6], [4]]), dtype=flow.int, device=flow.device(device)
    )
    of_update = flow.tensor(
        np.array([10.2, 5.1, 12.7]),
        requires_grad=True,
        dtype=flow.float,
        device=flow.device(device),
    )
    np_out = np.array([0.0, 10.2, 2.0, 3.0, 12.7, 5.0, 5.1, 7.0])
    np_update_grad = np.array([1.0, 1.0, 1.0])
    np_origin_grad = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    output = flow.tensor_scatter_nd_update(origin, indices, of_update)
    out_sum = output.sum()
    out_sum.backward()
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(of_update.grad.numpy(), np_update_grad))
    test_case.assertTrue(np.allclose(origin.grad.numpy(), np_origin_grad))


@flow.unittest.skip_unless_1n1d()
class TestTensorScatterNdUpdate(flow.unittest.TestCase):
    def test_tensor_scatter_nd_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_tensor_scatter_nd_update,
            _test_tensor_scatter_nd_update_with_non_contiguous_input,
            _test_tensor_scatter_nd_update_t,
            _test_tensor_scatter_nd_update_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
