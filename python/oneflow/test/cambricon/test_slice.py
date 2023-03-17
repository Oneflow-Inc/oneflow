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


def _test_slice_graph(test_case, device):
    np_arr = np.random.randn(4, 5, 6, 7).astype(np.float32)
    x = flow.tensor(np_arr, device=flow.device(device))

    @flow.nn.Graph.trace
    def test_func():
        return x[1:3, 2:4, ..., 1:5]

    out = test_func()
    test_case.assertTrue(np.array_equal(out.cpu().numpy(), np_arr[1:3, 2:4, ..., 1:5]))

    @flow.nn.Graph.trace
    def test_func():
        return x[0, ::1, ..., 2:3]

    out = test_func()
    test_case.assertTrue(np.array_equal(out.cpu().numpy(), np_arr[0, ::1, ..., 2:3]))


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
            _test_slice_graph,
        ]
        arg_dict["device"] = ["mlu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestSliceUpdate(flow.unittest.TestCase):
    def test_slice_update(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        input = flow.tensor(x).to("mlu")
        update = flow.tensor(np.array([2, 3, 4]).astype(np.float32)).to("mlu")
        output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
        flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
        test_case.assertTrue(np.array_equal(input.cpu().numpy(), output))

    def test_slice_update_negative_index(test_case):
        np_arr = np.zeros(shape=(2, 3, 4))
        input = flow.tensor(np_arr, dtype=flow.float32).to("mlu")
        np_arr[-1] = 1
        input[-1] = 1
        test_case.assertTrue(np.array_equal(input.cpu().numpy(), np_arr))

    def test_slice_update_scalar_integer_tensor_index(test_case):
        np_arr_a = np.random.rand(133, 1, 15)
        np_arr_b = np.random.rand(133, 2, 1)

        a_flow_cpu = flow.tensor(np_arr_a)
        b_flow_cpu = flow.tensor(np_arr_b)
        pos_flow_cpu = flow.tensor(0)
        a_flow_cpu[:, 0, pos_flow_cpu] = b_flow_cpu[:, 1, 0]

        a_flow = flow.tensor(np_arr_a).to("mlu")
        b_flow = flow.tensor(np_arr_b).to("mlu")
        pos_flow = flow.tensor(0)
        a_flow[:, 0, pos_flow] = b_flow[:, 1, 0]

        test_case.assertTrue(
            np.allclose(a_flow.cpu().numpy(), a_flow_cpu.numpy(), rtol=1e-5, atol=1e-5,)
        )

    def test_slice_update_scalar_boolean_tensor_index(test_case):
        np_arr_a = np.random.rand(2, 1, 2)
        np_arr_b = np.random.rand(2, 2, 1)

        a_flow_cpu = flow.tensor(np_arr_a)
        b_flow_cpu = flow.tensor(np_arr_b)
        pos_flow_cpu = flow.tensor(True)
        a_flow_cpu[:, 0, pos_flow_cpu] = b_flow_cpu[:, 1, 0]

        a_flow = flow.tensor(np_arr_a).to("mlu")
        b_flow = flow.tensor(np_arr_b).to("mlu")
        pos_flow = flow.tensor(True)
        a_flow[:, 0, pos_flow] = b_flow[:, 1, 0]

        test_case.assertTrue(
            np.allclose(a_flow.cpu().numpy(), a_flow_cpu.numpy(), rtol=1e-5, atol=1e-5,)
        )

    # TODO(): support expand op for mlu
    # def test_slice_update_negative_index_graph(test_case):
    #     np_arr = np.zeros(shape=(2, 3, 4))
    #     input = flow.tensor(np_arr, dtype=flow.float32).to("mlu")
    #     np_arr[-1] = 1

    #     @flow.nn.Graph.trace
    #     def test_func():
    #         input[-1] = 1
    #         return input

    #     out = test_func()
    #     test_case.assertTrue(np.array_equal(out.cpu().numpy(), np_arr))

    def test_slice_update_different_dtype(test_case):
        x = np.array([1, 1, 1, 1, 1]).astype(np.float32)
        for value_type in [np.int32, np.float16]:
            input = flow.tensor(x).to("mlu")
            update = flow.tensor(np.array([2, 3, 4]).astype(value_type)).to("mlu")
            output = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
            flow.slice_update(input, update, slice_tup_list=[[1, 4, 1]])
            test_case.assertTrue(np.array_equal(input.cpu().numpy(), output))

    def test_slice_update_ellipsis_type(test_case):
        np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
        input = flow.tensor(np_arr, dtype=flow.float32).to("mlu")
        np_arr[0, ::1, ..., 2:3] = 1
        input[0, ::1, ..., 2:3] = 1
        test_case.assertTrue(np.array_equal(input.cpu().numpy(), np_arr))

    # TODO(): support expand op for mlu
    # def test_slice_update_ellipsis_type_graph(test_case):
    #     np_arr = np.zeros(shape=(2, 3, 4, 5, 6))
    #     input = flow.tensor(np_arr, dtype=flow.float32).to("mlu")
    #     np_arr[0, ::1, ..., 2:3] = 1

    #     @flow.nn.Graph.trace
    #     def test_func():
    #         input[0, ::1, ..., 2:3] = 1
    #         return input

    #     out = test_func()
    #     test_case.assertTrue(np.array_equal(out.cpu().numpy(), np_arr))

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

    def test_slice_update_expand_value(test_case):
        ref_np = np.random.rand(2, 3, 4)
        ref_of = flow.tensor(ref_np).to("mlu")
        update_np = np.random.rand(3,)
        update_ref = flow.tensor(update_np).to("mlu")

        ref_of[:, :, 1] = update_ref
        ref_np[:, :, 1] = update_np
        test_case.assertTrue(np.array_equal(ref_of.cpu().numpy(), ref_np))


if __name__ == "__main__":
    unittest.main()
