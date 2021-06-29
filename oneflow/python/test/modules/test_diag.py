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

import oneflow.experimental as flow
from test_util import GenArgList

def _test_diag_one_dimen(test_case, device):
    input = flow.Tensor(np.random.randn(3),device = flow.device(device))
    of_out = flow.diag(input)
    np_out = np.diag(input.numpy())
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
    )


def _test_diag_one_dimen_positive(test_case, device):
    input = flow.Tensor(np.random.randn(3),device = flow.device(device))
    of_out = flow.diag(input, 1)
    np_out = np.diag(input.numpy(), 1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_diag_one_dimen_negative(test_case, device):
    input = flow.Tensor(np.random.randn(3),device = flow.device(device))
    of_out = flow.diag(input, -1)
    np_out = np.diag(input.numpy(), -1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3),device = flow.device(device))
    of_out = flow.diag(input)
    np_out = np.diag(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen_positive(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3),device = flow.device(device))
    of_out = flow.diag(input, 1)
    np_out = np.diag(input.numpy(), 1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen_negative(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3),device = flow.device(device))
    of_out = flow.diag(input, -1)
    np_out = np.diag(input.numpy(), -1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_diag_one_dimen_backward(test_case, device):
    input = flow.Tensor(np.random.randn(3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input).sum()
    of_out.backward()
    np_grad = np.ones(3)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


def _test_diag_one_dimen_backward_positive(test_case, device):
    input = flow.Tensor(np.random.randn(3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input, 1).sum()
    of_out.backward()
    np_grad = np.ones(shape = 3)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


def _test_diag_one_dimen_backward_negative(test_case, device):
    input = flow.Tensor(np.random.randn(3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input, -1).sum()
    of_out.backward()
    np_grad = np.ones(shape = 3)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen_backward(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input).sum()
    of_out.backward()
    np_grad = np.identity(3)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen_backward_positive(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input, 1).sum()
    of_out.backward()
    np_grad = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


def _test_diag_other_dimen_backward_negative(test_case, device):
    input = flow.Tensor(np.random.randn(3, 3), device = flow.device(device), requires_grad=True)
    of_out = flow.diag(input, -1).sum()
    of_out.backward()
    np_grad = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5, equal_nan=True))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)

class TestDiag(flow.unittest.TestCase):
    def test_diag(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_diag_one_dimen, 
            _test_diag_one_dimen_positive,
            _test_diag_one_dimen_negative,
            _test_diag_other_dimen,
            _test_diag_other_dimen_positive,
            _test_diag_other_dimen_negative,
            _test_diag_one_dimen_backward,
            _test_diag_one_dimen_backward_positive,
            _test_diag_one_dimen_backward_negative,
            _test_diag_other_dimen_backward,
            _test_diag_other_dimen_backward_positive,
            _test_diag_other_dimen_backward_negative,
            ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])
            

if __name__ == "__main__":
    unittest.main()
    