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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_diag_forward(test_case, shape, diagonal, device):
    input = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.diag(input, diagonal)
    np_out = np.diag(input.numpy(), diagonal)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
    )
    test_case.assertTrue(
        np.allclose(
            input.diag(diagonal=diagonal).numpy(), np_out, 1e-05, 1e-05, equal_nan=True
        )
    )


def _test_diag_one_dim_backward(test_case, diagonal, device):
    input = flow.Tensor(
        np.random.randn(3), device=flow.device(device), requires_grad=True
    )
    of_out = flow.diag(input, diagonal).sum()
    of_out.backward()
    np_grad = np.ones(shape=3)
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )
    input = flow.Tensor(
        np.random.randn(3), device=flow.device(device), requires_grad=True
    )
    of_out = input.diag(diagonal=diagonal).sum()
    of_out.backward()
    np_grad = np.ones(shape=3)
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )


def _test_diag_other_dim_backward(test_case, diagonal, device):
    input = flow.Tensor(
        np.random.randn(3, 3), device=flow.device(device), requires_grad=True
    )
    of_out = flow.diag(input, diagonal).sum()
    of_out.backward()
    if diagonal > 0:
        np_grad = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    elif diagonal < 0:
        np_grad = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    else:
        np_grad = np.identity(3)
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )
    input = flow.Tensor(
        np.random.randn(3, 3), device=flow.device(device), requires_grad=True
    )
    of_out = input.diag(diagonal=diagonal).sum()
    of_out.backward()
    if diagonal > 0:
        np_grad = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    elif diagonal < 0:
        np_grad = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    else:
        np_grad = np.identity(3)
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )


def _test_diag_other_dim_non_square_backward(test_case, diagonal, device):
    input = flow.Tensor(
        np.random.randn(3, 4), device=flow.device(device), requires_grad=True
    )
    of_out = flow.diag(input, diagonal).sum()
    of_out.backward()
    if diagonal > 0:
        np_tmp = np.zeros([3, 1])
        np_grad = np.identity(3)
        np_grad = np.hstack((np_tmp, np_grad))
    elif diagonal < 0:
        np_grad = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    else:
        np_tmp = np.zeros([3, 1])
        np_grad = np.identity(3)
        np_grad = np.hstack((np_grad, np_tmp))
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )
    input = flow.Tensor(
        np.random.randn(3, 4), device=flow.device(device), requires_grad=True
    )
    of_out = input.diag(diagonal=diagonal).sum()
    of_out.backward()
    if diagonal > 0:
        np_tmp = np.zeros([3, 1])
        np_grad = np.identity(3)
        np_grad = np.hstack((np_tmp, np_grad))
    elif diagonal < 0:
        np_grad = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    else:
        np_tmp = np.zeros([3, 1])
        np_grad = np.identity(3)
        np_grad = np.hstack((np_grad, np_tmp))
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, 1e-05, 1e-05, equal_nan=True)
    )


@flow.unittest.skip_unless_1n1d()
class TestDiag(flow.unittest.TestCase):
    def test_diag_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(3,), (3, 3), (3, 4)]
        arg_dict["diagonal"] = [1, 0, -1]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_diag_forward(test_case, *arg[0:])

    def test_diag_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_diag_one_dim_backward,
            _test_diag_other_dim_backward,
            _test_diag_other_dim_non_square_backward,
        ]
        arg_dict["diagonal"] = [1, 0, -1]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
