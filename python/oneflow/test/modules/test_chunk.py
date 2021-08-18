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


def _test_2_dim_forward(test_case, device):
    np_arr = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device))
    dim = 0
    chunks = 2
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(1, 3), (1, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 1
    chunks = 2
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(2, 1), (2, 2)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 1
    chunks = 3
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(2, 1), (2, 1), (2, 1)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))


def _test_2_dim_tensor_function_forward(test_case, device):
    np_arr = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device))
    dim = 0
    chunks = 2
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(1, 3), (1, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 1
    chunks = 2
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(2, 1), (2, 2)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 1
    chunks = 3
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(2, 1), (2, 1), (2, 1)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))


def _test_4_dim_forward(test_case, device):
    np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device))
    dim = 2
    chunks = 3
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 2
    chunks = 4
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(5, 3, 1, 9), (5, 3, 1, 9), (5, 3, 1, 9), (5, 3, 3, 9)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 3
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(5, 3, 6, 3), (5, 3, 6, 3), (5, 3, 6, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 2
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(5, 3, 6, 4), (5, 3, 6, 5)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 4
    of_out = flow.chunk(input, chunks, dim)
    np_out_shape = [(5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))


def _test_4_dim_tensor_function_forward(test_case, device):
    np_arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device))
    dim = 2
    chunks = 3
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 2
    chunks = 4
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(5, 3, 1, 9), (5, 3, 1, 9), (5, 3, 1, 9), (5, 3, 3, 9)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 3
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(5, 3, 6, 3), (5, 3, 6, 3), (5, 3, 6, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 2
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(5, 3, 6, 4), (5, 3, 6, 5)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))
    dim = 3
    chunks = 4
    of_out = input.chunk(chunks, dim)
    np_out_shape = [(5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 3)]
    for i in range(0, chunks):
        of_out_shape = of_out[i].numpy().shape
        test_case.assertTrue(np.allclose(of_out_shape, np_out_shape[i], 1e-05, 1e-05))


def _test_chunk_backward(test_case, device):
    np_arr = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_arr, device=flow.device(device))
    input.requires_grad = True
    y = flow.chunk(input, chunks=2, dim=0)
    (z1, z2) = (y[0].sum(), y[1].sum())
    z1.backward()
    z2.backward()
    np_grad = np.ones((2, 3))
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@flow.unittest.skip_unless_1n1d()
class TestChunk(flow.unittest.TestCase):
    def test_chunk(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_2_dim_forward,
            _test_4_dim_forward,
            _test_2_dim_tensor_function_forward,
            _test_4_dim_tensor_function_forward,
            _test_chunk_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
