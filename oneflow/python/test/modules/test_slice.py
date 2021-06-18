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


def _test_slice(test_case, device):
    np_arr = np.random.randn(3, 6, 9).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    tmp = np_arr[0:3, 0:5, 0:6]
    np_out = tmp[::1, ::2, ::3]
    test_case.assertTrue(np.array_equal(y.numpy(), np_out))


def _test_slice_1_dim(test_case, device):
    np_arr = np.random.randn(100).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    test_case.assertTrue(np.allclose(x[1].numpy(), np_arr[1], 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(x[99].numpy(), np_arr[99], 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(x[0:2].numpy(), np_arr[0:2], 1e-5, 1e-5))


def _test_slice_3_dim(test_case, device):
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    test_case.assertTrue(np.allclose(x[:, 0].numpy(), np_arr[:, 0], 1e-5, 1e-5))


def _test_slice_4_dim(test_case, device):
    np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    tup_list = [[0, 5, 2], [None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    tmp = np_arr[0:5, 0:3, 0:5, 0:6]
    np_out = tmp[::2, ::1, ::2, ::3]
    test_case.assertTrue(np.array_equal(y.numpy(), np_out))


def _test_slice_with_int_index(test_case, device):
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    of_out = x[0, 1:2]
    np_out = np_arr[0, 1:2]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    of_out = x[0, :]
    np_out = np_arr[0, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    of_out = x[0, :, :]
    np_out = np_arr[0, :, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))
    of_out = x[0, :, :, :]
    np_out = np_arr[0, :, :, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_slice_ellipsis_type(test_case, device):
    np_arr = np.random.randn(2, 3, 4, 5, 6, 7).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device))

    of_out = x[..., ::2, ::2, 3:4]
    np_out = np_arr[..., ::2, ::2, 3:4]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    of_out = x[..., 1:2, ::2, 1, ::3]
    np_out = np_arr[..., 1:2, ::2, 1, ::3]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    of_out = x[0, 2, ..., 1, 1:2]
    np_out = np_arr[0, 2, ..., 1, 1:2]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    of_out = x[::2, ..., 1:2]
    np_out = np_arr[::2, ..., 1:2]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_slice_backward(test_case, device):
    np_arr = np.random.randn(3, 6, 9).astype(np.float32)
    x = flow.Tensor(np_arr, device=flow.device(device), requires_grad=True)
    tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    z = y.sum()
    z.backward()

    np_grad = np.zeros((3, 6, 9))
    np_grad[0:3, 0:5, 0:6][::1, ::2, ::3] = 1
    test_case.assertTrue(np.array_equal(x.grad.numpy(), np_grad))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSlice(flow.unittest.TestCase):
    def test_slice(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_slice,
            _test_slice_1_dim,
            _test_slice_3_dim,
            _test_slice_4_dim,
            _test_slice_with_int_index,
            _test_slice_ellipsis_type,
            _test_slice_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSliceUpdate(flow.unittest.TestCase):
    def test_slice_update(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.Tensor(x)
        update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        y = flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
        test_case.assertTrue(np.array_equal(y.numpy(), output))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLogicalSliceAssign(flow.unittest.TestCase):
    # this is an in-place operation, so requires_grad should be False(no grad in backward)
    def test_logical_slice_assign(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.Tensor(x)
        update = flow.Tensor(np.array([2, 3, 4]).astype(np.float32))
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        flow.tmp.logical_slice_assign(input, update, slice_tup_list=[[1, 4, 1]])
        test_case.assertTrue(np.array_equal(input.numpy(), output))

    def test_logical_slice_assign_ellipsis_type(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.Tensor(np_arr)
        np_arr[0, ::1, ..., 2:3] = 1
        input[0, ::1, ..., 2:3] = 1
        test_case.assertTrue(np.array_equal(input.numpy(), np_arr))


if __name__ == "__main__":
    unittest.main()
