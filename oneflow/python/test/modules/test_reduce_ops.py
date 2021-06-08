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


def _test_min(test_case, device, shape, dim, keepdims):
    input_arr = np.random.randn(*shape)
    np_out = np.amin(input_arr, axis=dim, keepdims=keepdims)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.min(x, dim, keepdims)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()

    np_out_grad = np.zeros_like(input_arr)
    if dim == None:
        arg_min = np.argmin(input_arr)
        np.put(np_out_grad, arg_min, 1)
    else:
        arg_min = np.expand_dims(np.argmin(input_arr, axis=dim), axis=dim)
        np.put_along_axis(np_out_grad, arg_min, 1, axis=dim)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4))


def _test_min_tensor_function(test_case, device, shape, dim, keepdims):
    input_arr = np.random.randn(*shape)
    np_out = np.amin(input_arr, axis=dim, keepdims=keepdims)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = x.min(dim, keepdims)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()

    np_out_grad = np.zeros_like(input_arr)
    if dim == None:
        arg_min = np.argmin(input_arr)
        np.put(np_out_grad, arg_min, 1)
    else:
        arg_min = np.expand_dims(np.argmin(input_arr, axis=dim), axis=dim)
        np.put_along_axis(np_out_grad, arg_min, 1, axis=dim)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMinModule(flow.unittest.TestCase):
    def test_max(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_min, _test_min_tensor_function]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4, 5)]
        arg_dict["dim"] = [None, 0, -1]
        arg_dict["keepdims"] = [False, True]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


def _test_max(test_case, device, shape, dim, keepdims):
    input_arr = np.random.randn(*shape)
    np_out = np.amax(input_arr, axis=dim, keepdims=keepdims)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.max(x, dim, keepdims)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()

    np_out_grad = np.zeros_like(input_arr)
    if dim == None:
        arg_max = np.argmax(input_arr)
        np.put(np_out_grad, arg_max, 1)
    else:
        arg_max = np.expand_dims(np.argmax(input_arr, axis=dim), axis=dim)
        np.put_along_axis(np_out_grad, arg_max, 1, axis=dim)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4))


def _test_max_tensor_function(test_case, device, shape, dim, keepdims):
    input_arr = np.random.randn(*shape)
    np_out = np.amax(input_arr, axis=dim, keepdims=keepdims)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = x.max(dim, keepdims)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()

    np_out_grad = np.zeros_like(input_arr)
    if dim == None:
        arg_max = np.argmax(input_arr)
        np.put(np_out_grad, arg_max, 1)
    else:
        arg_max = np.expand_dims(np.argmax(input_arr, axis=dim), axis=dim)
        np.put_along_axis(np_out_grad, arg_max, 1, axis=dim)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMaxModule(flow.unittest.TestCase):
    def test_max(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_max, _test_max_tensor_function]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4, 5)]
        arg_dict["dim"] = [None, 0, -1]
        arg_dict["keepdims"] = [False, True]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
