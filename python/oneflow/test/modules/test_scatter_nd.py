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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_scatter_nd(test_case, device):
    indices = flow.tensor(
        np.array([[1], [6], [4]]), dtype=flow.int, device=flow.device(device)
    )
    update = flow.tensor(
        np.array([10.2, 5.1, 12.7]), dtype=flow.float, device=flow.device(device)
    )
    np_out = np.array([0.0, 10.2, 0.0, 0.0, 12.7, 0.0, 5.1, 0.0])
    output = flow.scatter_nd(indices, update, [8])
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_scatter_nd_t(test_case, device):
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
            [0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )
    output = flow.scatter_nd(indices, update, [5, 3])
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))


def _test_scatter_nd_backward(test_case, device):
    indices = flow.tensor(
        np.array([[1], [6], [4]]), dtype=flow.int, device=flow.device(device)
    )
    of_update = flow.tensor(
        np.array([10.2, 5.1, 12.7]),
        requires_grad=True,
        dtype=flow.float,
        device=flow.device(device),
    )
    np_out = np.array([0.0, 10.2, 0.0, 0.0, 12.7, 0.0, 5.1, 0.0])
    np_grad = np.array([1.0, 1.0, 1.0])
    output = flow.scatter_nd(indices, of_update, [8])
    out_sum = output.sum()
    out_sum.backward()
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 0.0001, 0.0001))
    test_case.assertTrue(np.array_equal(of_update.grad.numpy(), np_grad))


@flow.unittest.skip_unless_1n1d()
class TestScatter_nd(flow.unittest.TestCase):
    def test_scatter_nd(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_scatter_nd,
            _test_scatter_nd_t,
            _test_scatter_nd_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
