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
from oneflow.test_utils.automated_test_util import util

import oneflow as flow
import oneflow.unittest


def _test_slice(test_case, device):
    np_arr = np.random.randn(3, 6, 9).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    flow_tmp = x[0:3, 0:5, 0:6]
    y = flow_tmp[::1, ::2, ::3]
    tmp = np_arr[0:3, 0:5, 0:6]
    np_out = tmp[::1, ::2, ::3]
    test_case.assertTrue(np.array_equal(y.numpy(), np_out))


def _test_slice_empty(test_case, device):
    np_arr = np.random.randn(10).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    y = x[3:3]
    test_case.assertTrue(y.shape, flow.Size((0,)))
    np_out = np_arr[3:3]
    test_case.assertTrue(np.array_equal(y.numpy(), np_out))


def _test_slice_1_dim(test_case, device):
    np_arr = np.random.randn(100).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    test_case.assertTrue(np.allclose(x[1].numpy(), np_arr[1], 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(x[99].numpy(), np_arr[99], 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(x[0:2].numpy(), np_arr[0:2], 1e-05, 1e-05))


def _test_slice_3_dim(test_case, device):
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    test_case.assertTrue(np.allclose(x[:, 0].numpy(), np_arr[:, 0], 1e-05, 1e-05))


def _test_slice_4_dim(test_case, device):
    np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    tup_list = [[0, 5, 2], [None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    tmp = np_arr[0:5, 0:3, 0:5, 0:6]
    np_out = tmp[::2, ::1, ::2, ::3]
    test_case.assertTrue(np.array_equal(y.numpy(), np_out))


def _test_slice_with_int_index(test_case, device):
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    of_out = x[0, 1:2]
    np_out = np_arr[0, 1:2]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    of_out = x[0, :]
    np_out = np_arr[0, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    of_out = x[0, :, :]
    np_out = np_arr[0, :, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))
    np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
    of_out = x[0, :, :, :]
    np_out = np_arr[0, :, :, :]
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_slice_negative_index(test_case, device):
    np_arr = np.random.randn(4, 5, 6)
    x = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    test_case.assertTrue(np.allclose(x[-1].numpy(), np_arr[-1], 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(x[-2].numpy(), np_arr[-2], 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(x[-3].numpy(), np_arr[-3], 0.0001, 0.0001))
    test_case.assertTrue(np.allclose(x[-4].numpy(), np_arr[-4], 0.0001, 0.0001))


def _test_slice_ellipsis_type(test_case, device):
    np_arr = np.random.randn(2, 3, 4, 5, 6, 7).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))
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
    x = flow.tensor(np_arr, device=flow.device(device), requires_grad=True)
    tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
    y = flow.slice(x, slice_tup_list=tup_list)
    z = y.sum()
    z.backward()
    np_grad = np.zeros((3, 6, 9))
    np_grad[0:3, 0:5, 0:6][::1, ::2, ::3] = 1
    test_case.assertTrue(np.array_equal(x.grad.numpy(), np_grad))


def _test_slice_update(test_case, device):
    x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
    input = flow.tensor(x, requires_grad=True)
    input.retain_grad()
    update = flow.tensor(np.array([2, 3, 4]).astype(np.float32), requires_grad=True)
    output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
    # Get the inplaced tensor grad by another tensor
    t = input + 0
    flow._C.slice_update(t, update, [1,], [4,], [1,], inplace=True)
    z = t.sum()
    z.backward()
    test_case.assertTrue(np.array_equal(t.numpy(), output))
    np_grad = np.zeros(x.shape)
    np_grad[0] = 1
    np_grad[4] = 1
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))
    test_case.assertTrue(np.array_equal(update.grad.numpy(), np.ones(update.shape)))


def _test_slice_update_with_stride(test_case, device):
    arr = np.arange(24).reshape(2, 2, 2, 3).astype(np.float32)
    np_in = arr
    np_out = np_in.transpose(1, 0, 2, 3)
    np_out[0:1, 1:2, :, 1:2] = 3.1415

    input = flow.tensor(arr, device=flow.device(device))
    output = input.permute(1, 0, 2, 3)
    output[0:1, 1:2, :, 1:2] = 3.1415

    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


@flow.unittest.skip_unless_1n1d()
class TestSlice(flow.unittest.TestCase):
    def test_slice(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_slice,
            _test_slice_empty,
            _test_slice_1_dim,
            _test_slice_3_dim,
            _test_slice_4_dim,
            _test_slice_with_int_index,
            _test_slice_negative_index,
            _test_slice_ellipsis_type,
            _test_slice_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestSliceUpdate(flow.unittest.TestCase):
    def test_slice(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_slice_update,
            # # TODO:(zhaoluyang) test when slice_update support stride
            # _test_slice_update_with_stride
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_slice_update_graph(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x, requires_grad=True)
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32), requires_grad=True)
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])

        class TestModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = flow.nn.Parameter(flow.Tensor(x))

            def forward(self, x, update):
                flow._C.slice_update(x, update, [1,], [4,], [1,], inplace=True)
                y = x + self.weight
                return x, y

        test_m = TestModule()
        of_sgd = flow.optim.SGD(test_m.parameters(), lr=0.001, momentum=0.9)

        class TestSliceUpdateGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = test_m
                self.add_optimizer(of_sgd)

            def build(self, x, update):
                x, y = self.m(x, update)
                z = y.sum()
                z.backward()
                return x

        slice_update_g = TestSliceUpdateGraph()

        y = slice_update_g(input, update)
        test_case.assertTrue(np.array_equal(y.numpy(), output))
        # TODO(): check grad of slice_update in graph.


@flow.unittest.skip_unless_1n1d()
class TestLogicalSliceAssign(flow.unittest.TestCase):
    def test_logical_slice_assign(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x)
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32))
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        flow.logical_slice_assign(input, update, slice_tup_list=[[1, 4, 1]])
        test_case.assertTrue(np.array_equal(input.numpy(), output))

    def test_logical_slice_assign_graph(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x)
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32))
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])

        @flow.nn.Graph.to_graph
        def test_func(input):
            flow.logical_slice_assign(input, update, slice_tup_list=[[1, 4, 1]])
            return input

        # NOTE(strint): input outside the graph has not been change yet currently.
        out = test_func(input)
        test_case.assertTrue(np.array_equal(out.numpy(), output))

    def test_logical_slice_assign_negative_index(test_case):
        np_arr = np.zeros(shape=(2, 3, 4))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[-1] = 1
        input[-1] = 1
        test_case.assertTrue(np.array_equal(input.numpy(), np_arr))

    def test_logical_slice_assign_negative_index_graph(test_case):
        np_arr = np.zeros(shape=(2, 3, 4))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[-1] = 1

        @flow.nn.Graph.to_graph
        def test_func():
            input[-1] = 1
            return input

        out = test_func()
        test_case.assertTrue(np.array_equal(out.numpy(), np_arr))

    def test_logical_slice_assign_ellipsis_type(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[0, ::1, ..., 2:3] = 1
        input[0, ::1, ..., 2:3] = 1
        test_case.assertTrue(np.array_equal(input.numpy(), np_arr))

    def test_logical_slice_assign_ellipsis_type_graph(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[0, ::1, ..., 2:3] = 1

        @flow.nn.Graph.to_graph
        def test_func():
            input[0, ::1, ..., 2:3] = 1
            return input

        out = test_func()
        test_case.assertTrue(np.array_equal(out.numpy(), np_arr))


if __name__ == "__main__":
    unittest.main()
