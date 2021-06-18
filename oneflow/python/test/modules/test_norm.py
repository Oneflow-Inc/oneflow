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


def _np_vector_norm_backward(x, ord=2, dim=None):
    re = np.zeros_like(x)
    if isinstance(ord, int) and isinstance(dim, int):
        if ord == 0:
            return re
        else:
            temp = (np.sum(np.abs(x ** ord), dim)) ** (1.0 / ord - 1)
            re = np.where(x ** ord < 0, -temp, temp) * (x ** (ord - 1))

    elif dim == None and x.ndim == 1:
        if ord == 0:
            return re
        elif ord == float("inf"):
            max_ind = np.argmax(np.abs(x))
            re[max_ind] += 1 if x[max_ind] != 0 else 0
            re = np.where(x < 0, -re, re)
        elif ord == float("-inf"):
            min_ind = np.argmin(np.abs(x))
            re[min_ind] += 1 if x[min_ind] != 0 else 0
            re = np.where(x < 0, -re, re)
        else:
            temp = (np.sum(np.abs(x ** ord))) ** (1.0 / ord - 1)
            re = np.where(x ** ord < 0, -temp, temp) * (x ** (ord - 1))

    elif (
        isinstance(ord, float)
        and isinstance(dim, int)
        and ord in [float("inf"), float("-inf")]
    ):
        if ord == float("inf"):
            max_ind = np.argmax(np.abs(x), dim)
            index = (
                [(i, max_ind[i]) for i in range(len(max_ind))]
                if dim == 1
                else [(max_ind[i], i) for i in range(len(max_ind))]
            )
            print(index)
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)
        else:
            min_ind = np.argmin(np.abs(x), dim)
            index = (
                [(i, min_ind[i]) for i in range(len(min_ind))]
                if dim == 1
                else [(min_ind[i], i) for i in range(len(min_ind))]
            )
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)
    return re


def _np_matrix_norm_backward(x, ord="fro"):
    re = np.zeros_like(x)

    if isinstance(ord, int):
        if ord == 1:
            max_ind = np.argmax(np.sum(np.abs(x), 0))
            index = [(i, max_ind) for i in range(x.shape[0])]
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)
        elif ord == -1:
            min_ind = np.argmin(np.sum(np.abs(x), 0))
            index = [(i, min_ind) for i in range(x.shape[0])]
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)

    elif ord == "fro":
        re = (np.sum(x ** 2)) ** (-0.5) * x

    elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
        if ord == float("inf"):
            max_ind = np.argmax(np.sum(np.abs(x), 1))
            index = [(max_ind, i) for i in range(x.shape[1])]
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)
        else:
            min_ind = np.argmin(np.sum(np.abs(x), 1))
            index = [(min_ind, i) for i in range(x.shape[1])]
            for j in index:
                re[j] += 1 if x[j] != 0 else 0
            re = np.where(x < 0, -re, re)
    return re


def _test_norm_1d(test_case, device):
    input = flow.Tensor(
        np.random.randn(10,), dtype=flow.float32, device=flow.device(device)
    )
    of_out_1 = flow.linalg.norm(input)
    of_out_2 = flow.linalg.norm(input, ord=0)
    of_out_3 = flow.linalg.norm(input, ord=3)
    of_out_4 = flow.linalg.norm(input, ord=float("inf"))
    of_out_5 = flow.linalg.norm(input, ord=-float("inf"))
    np_out_1 = np.linalg.norm(input.numpy())
    np_out_2 = np.linalg.norm(input.numpy(), ord=0)
    np_out_3 = np.linalg.norm(input.numpy(), ord=3)
    np_out_4 = np.linalg.norm(input.numpy(), ord=float("inf"))
    np_out_5 = np.linalg.norm(input.numpy(), ord=-float("inf"))
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_4.numpy(), np_out_4, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_5.numpy(), np_out_5, 1e-5, 1e-5))


def _test_norm_2d(test_case, device):
    input = flow.Tensor(
        np.random.randn(5, 4), dtype=flow.float32, device=flow.device(device)
    )
    of_out_1 = flow.linalg.norm(input)
    of_out_2 = flow.linalg.norm(input, dim=0)
    of_out_3 = flow.linalg.norm(input, dim=1, keepdim=True)
    of_out_4 = flow.linalg.norm(input, ord=1, dim=0)
    of_out_5 = flow.linalg.norm(input, ord=-1, dim=1, keepdim=True)
    np_out_1 = np.linalg.norm(input.numpy())
    np_out_2 = np.linalg.norm(input.numpy(), axis=0)
    np_out_3 = np.linalg.norm(input.numpy(), axis=1, keepdims=True)
    np_out_4 = np.linalg.norm(input.numpy(), ord=1, axis=0)
    np_out_5 = np.linalg.norm(input.numpy(), ord=-1, axis=1, keepdims=True)
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_4.numpy(), np_out_4, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_5.numpy(), np_out_5, 1e-5, 1e-5))


def _test_norm_Nd(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(3, 4, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(3, 4, 3, 5), dtype=flow.float32, device=flow.device(device)
    )
    of_out_1 = flow.linalg.norm(input1)
    of_out_2 = flow.linalg.norm(input1, dim=(0, 1))
    of_out_3 = flow.linalg.norm(input2, dim=(0, 2))

    np_out_1 = np.linalg.norm(input1.numpy())
    np_out_2 = np.linalg.norm(input1.numpy(), axis=(0, 1))
    np_out_3 = np.linalg.norm(input2.numpy(), axis=(0, 2))
    test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-5, 1e-5))


def _test_fro_order_norm_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(5, 4),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.linalg.norm(input)
    of_out.backward()
    np_out_grad = _np_matrix_norm_backward(input.numpy())
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_out_grad, 1e-5, 1e-5))


def _test_1d_inf_order_norm_backward(test_case, device):
    for ord in [float("inf"), -float("inf")]:
        input = flow.Tensor(
            np.random.randn(5,),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.linalg.norm(input, ord=ord)
        of_out.backward()
        np_out_grad = _np_vector_norm_backward(input.numpy(), ord=ord)
        test_case.assertTrue(np.allclose(input.grad.numpy(), np_out_grad, 1e-5, 1e-5))


def _test_2d_inf_order_norm_backward(test_case, device):
    for ord in [float("inf"), -float("inf")]:
        input = flow.Tensor(
            np.random.randn(5, 4),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.linalg.norm(input, ord=ord)
        of_out.backward()
        np_out_grad = _np_matrix_norm_backward(input.numpy(), ord=ord)
        test_case.assertTrue(np.allclose(input.grad.numpy(), np_out_grad, 1e-5, 1e-5))


def _test_1d_digits_order_norm_backward(test_case, device):
    for ord in [1, -1, 2, -2, 5]:
        input = flow.Tensor(
            np.random.randn(5,),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.linalg.norm(input, ord=ord)
        of_out.backward()
        np_out_grad = _np_vector_norm_backward(input.numpy(), ord=ord)
        test_case.assertTrue(np.allclose(input.grad.numpy(), np_out_grad, 1e-5, 1e-5))


def _test_2d_digits_order_norm_backward(test_case, device):
    for ord in [1, -1]:
        input = flow.Tensor(
            np.random.randn(4, 5),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.linalg.norm(input, ord=ord)
        of_out.backward()
        np_out_grad = _np_matrix_norm_backward(input.numpy(), ord=ord)
        test_case.assertTrue(np.allclose(input.grad.numpy(), np_out_grad, 1e-5, 1e-5))


class TestNormModule(flow.unittest.TestCase):
    def test_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_norm_1d,
            _test_norm_2d,
            _test_norm_Nd,
            _test_fro_order_norm_backward,
            _test_1d_inf_order_norm_backward,
            _test_2d_inf_order_norm_backward,
            _test_1d_digits_order_norm_backward,
            _test_2d_digits_order_norm_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
