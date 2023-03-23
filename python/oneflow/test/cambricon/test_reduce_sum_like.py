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


def _get_np_dtype(oneflow_dtype):
    if oneflow_dtype == flow.float32:
        return np.float32
    if oneflow_dtype == flow.float16:
        return np.float16
    if oneflow_dtype == flow.int32:
        return np.int32

def _test_reduce_sum_like(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(3, 3, 3), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(3, 1, 1), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(1, 2))
    
    np_out = np.full(shape=like_tensor.shape, fill_value=9, dtype=np_dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_one(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(1, 2, 3), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(1, 1), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(1, 2))
    np_out = np.full(like_tensor.shape, 6)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_different_dim(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(2, 3, 4), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(3, 1), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(0, 2))
    np_out = np.full(like_tensor.shape, 8)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_different_dim_with_input_axisvec(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(1, 5, 6, 1, 6), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(1, 5, 6), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(3, 4))
    np_out = np.full(like_tensor.shape, 6)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_3dim(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(3, 3, 2), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(1, 3, 2), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(0,))
    np_out = np.full(like_tensor.shape, 3)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_4dim(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(3, 3, 2, 3), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(1, 3, 2, 1), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    of_out = flow._C.reduce_sum_like(input, like_tensor, axis=(0, 3))
    np_out = np.full(like_tensor.shape, 9)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_reduce_sum_like_empty_axis(test_case, device, dtype):
    np_dtype = _get_np_dtype(dtype)
    input = flow.tensor(
        np.ones(shape=(3, 3, 3), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    like_tensor = flow.tensor(
        np.ones(shape=(3, 3, 3), dtype=np_dtype),
        dtype=dtype,
        device=flow.device(device),
    )
    input_cpu = flow.tensor(
        np.ones(shape=(3, 3, 3), dtype=np_dtype), dtype=dtype, device="cpu",
    )
    like_tensor_cpu = flow.tensor(
        np.ones(shape=(3, 3, 3), dtype=np_dtype), dtype=dtype, device="cpu",
    )
    mlu_out = flow._C.reduce_sum_like(input, like_tensor, axis=())
    cpu_out = flow._C.reduce_sum_like(input_cpu, like_tensor_cpu, axis=())
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestReduceSumLikeModule(flow.unittest.TestCase):
    def test_reduce_sum_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_reduce_sum_like,
            _test_reduce_sum_like_one,
            _test_reduce_sum_like_different_dim,
            _test_reduce_sum_like_different_dim_with_input_axisvec,
            _test_reduce_sum_like_3dim,
            _test_reduce_sum_like_4dim,
            _test_reduce_sum_like_empty_axis,
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float32, flow.int32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
