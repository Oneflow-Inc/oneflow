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
from random import randint

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


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
    def test_slice_update(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x)
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32))
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
        test_case.assertTrue(np.array_equal(input.numpy(), output))

    def test_slice_update_negative_index(test_case):
        np_arr = np.zeros(shape=(2, 3, 4))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[-1] = 1
        input[-1] = 1
        test_case.assertTrue(np.array_equal(input.numpy(), np_arr))

    def test_slice_update_scalar_integer_tensor_index(test_case):
        np_arr_a = np.random.rand(133, 1, 15)
        np_arr_b = np.random.rand(133, 2, 1)

        a_torch = torch.tensor(np_arr_a)
        b_torch = torch.tensor(np_arr_b)
        pos_torch = torch.tensor(0)
        a_torch[:, 0, pos_torch] = b_torch[:, 1, 0]

        a_flow = flow.tensor(np_arr_a)
        b_flow = flow.tensor(np_arr_b)
        pos_flow = flow.tensor(0)
        a_flow[:, 0, pos_flow] = b_flow[:, 1, 0]

        test_case.assertTrue(
            np.allclose(a_flow.numpy(), a_torch.cpu().numpy(), rtol=1e-5, atol=1e-5,)
        )

    def test_slice_update_scalar_boolean_tensor_index(test_case):
        np_arr_a = np.random.rand(2, 1, 2)
        np_arr_b = np.random.rand(2, 2, 1)

        a_torch = torch.tensor(np_arr_a)
        b_torch = torch.tensor(np_arr_b)
        pos_torch = torch.tensor(True)
        a_torch[:, 0, pos_torch] = b_torch[:, 1, 0]

        a_flow = flow.tensor(np_arr_a)
        b_flow = flow.tensor(np_arr_b)
        pos_flow = flow.tensor(True)
        a_flow[:, 0, pos_flow] = b_flow[:, 1, 0]

        test_case.assertTrue(
            np.allclose(a_flow.numpy(), a_torch.cpu().numpy(), rtol=1e-5, atol=1e-5,)
        )

    def test_slice_update_negative_index_graph(test_case):
        np_arr = np.zeros(shape=(2, 3, 4))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[-1] = 1

        @flow.nn.Graph.trace
        def test_func():
            input[-1] = 1
            return input

        out = test_func()
        test_case.assertTrue(np.array_equal(out.numpy(), np_arr))

    def test_slice_update_different_dtype(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        for value_type in [np.int32, np.float64]:
            input = flow.tensor(x)
            update = flow.tensor(np.array([2, 3, 4]).astype(value_type))
            output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
            flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
            test_case.assertTrue(np.array_equal(input.numpy(), output))

    def test_slice_update_ellipsis_type(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[0, ::1, ..., 2:3] = 1
        input[0, ::1, ..., 2:3] = 1
        test_case.assertTrue(np.array_equal(input.numpy(), np_arr))

    def test_slice_update_ellipsis_type_graph(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.tensor(np_arr, dtype=flow.float32)
        np_arr[0, ::1, ..., 2:3] = 1

        @flow.nn.Graph.trace
        def test_func():
            input[0, ::1, ..., 2:3] = 1
            return input

        out = test_func()
        test_case.assertTrue(np.array_equal(out.numpy(), np_arr))

    def test_slice_update_grad_graph(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x, requires_grad=True)
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32), requires_grad=True)
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])

        class TestModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.ref_grad = flow.nn.Parameter(flow.zeros(5))
                self.value_grad = flow.nn.Parameter(flow.zeros(3))

            def forward(self, ref, value):
                x = ref + self.ref_grad
                y = value + self.value_grad
                return flow._C.slice_update(x, y, [1,], [4,], [1,])

        test_m = TestModule()
        of_sgd = flow.optim.SGD(test_m.parameters(), lr=1.0, momentum=0.0)

        class TestSliceUpdateGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = test_m
                self.add_optimizer(of_sgd)

            def build(self, ref, update):
                x = self.m(ref, update)
                x.sum().backward()
                return x

        slice_update_g = TestSliceUpdateGraph()

        y = slice_update_g(input, update)

        # forward
        test_case.assertTrue(np.array_equal(y.numpy(), output))
        # ref grad
        ref_grad = np.array([1.0, 0.0, 0.0, 0.0, 1.0]).astype(np.float32)
        test_case.assertTrue(np.array_equal(-test_m.ref_grad, ref_grad))
        # value grad
        value_grad = np.array([1.0, 1.0, 1.0]).astype(np.float32)
        test_case.assertTrue(np.array_equal(-test_m.value_grad, value_grad))

    def test_random_nd_slice_update_in_non_contiguous_tensor(test_case):
        def get_random_slice_tuple(shape):
            slice_tup = []
            slice_size = []
            for i in range(len(shape)):
                start = randint(0, shape[i] - 1)
                end = randint(start + 1, shape[i])
                step = randint(1, end - start + 1)
                slice_tup.append(slice(start, end, step))
                slice_size.append((end - start + step - 1) // step)
            return tuple(slice_tup), tuple(slice_size)

        def get_random_update_shape_and_perm(shape):
            perm = flow.randperm(len(shape)).tolist()
            no_perm_shape = [shape[i] for i in perm]
            inv_perm = [0] * len(shape)
            for i in range(len(shape)):
                inv_perm[perm[i]] = i
            return no_perm_shape, inv_perm

        def compare_result_between_oneflow_and_numpy(test_case, shape):
            device = random_device().value()
            # non-contiguous ref
            ref = (
                flow.rand(shape, dtype=flow.float32)
                .to(device)
                .permute(flow.randperm(len(shape)).tolist())
            )
            ref_np = ref.detach().clone().numpy()
            shape = ref.shape
            # slice param
            slice_tup, slice_size = get_random_slice_tuple(shape)
            # non-contiguous update
            no_perm_shape, perm = get_random_update_shape_and_perm(slice_size)
            update = (
                flow.rand(no_perm_shape, dtype=flow.float32).to(device).permute(perm)
            )
            update_np = update.detach().clone().numpy()

            ref_np[slice_tup] = update_np
            # non-inplace update
            # NOTE: should test non-inplace first
            def slice_tuple_to_slice_list(slice_tup):
                # NOTE: oneflow.slice_update don't support passing slice parameters.
                slice_list = []
                for i in range(len(slice_tup)):
                    slice_list.append(
                        (slice_tup[i].start, slice_tup[i].stop, slice_tup[i].step)
                    )
                return slice_list

            of_res = flow.slice_update(
                ref, update, slice_tuple_to_slice_list(slice_tup)
            )
            test_case.assertTrue(np.array_equal(of_res.numpy(), ref_np))
            # inplace update
            ref[slice_tup] = update
            test_case.assertTrue(np.array_equal(ref.numpy(), ref_np))

        for dims in (2, 3, 4):
            for _ in range(10):
                shape = [randint(1, 21) for _ in range(dims)]
                compare_result_between_oneflow_and_numpy(test_case, shape)

    def test_slice_update_expand_value(test_case):
        ref_np = np.random.rand(2, 3, 4)
        ref_of = flow.tensor(ref_np)
        update_np = np.random.rand(3,)
        update_ref = flow.tensor(update_np)

        ref_of[:, :, 1] = update_ref
        ref_np[:, :, 1] = update_np
        test_case.assertTrue(np.array_equal(ref_of.numpy(), ref_np))


if __name__ == "__main__":
    unittest.main()
