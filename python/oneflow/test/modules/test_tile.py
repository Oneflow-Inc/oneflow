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


def np_tile(x, sizes):
    return np.tile(x, sizes)


def np_tile_grad(x, sizes):
    times = np.array(sizes).prod()
    return np.ones(shape=x.shape) * times


def _test_tile_less_dim_a(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 4, 1, 3), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (2,)
    np_out = np_tile(input.numpy(), sizes)
    of_out = input.tile(reps=sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tile_less_dim_b(test_case, device):
    input = flow.Tensor(
        np.random.randn(3, 2, 5), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (3, 4)
    np_out = np_tile(input.numpy(), sizes)
    of_out = input.tile(reps=sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tile_less_dim_c(test_case, device):
    input = flow.Tensor(
        np.random.randn(4, 3, 2, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (2, 3, 4, 4)
    np_out = np_tile(input.numpy(), sizes)
    of_out = input.tile(reps=sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tile_same_dim(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    sizes = (4, 2, 3, 19)
    of_out = input.tile(reps=sizes)
    np_out = np_tile(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tile_same_dim_int(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3), dtype=flow.int32, device=flow.device(device)
    )
    size_tensor = flow.Tensor(np.random.randn(4, 2, 3, 19))
    sizes = size_tensor.size()
    of_out = input.tile(reps=sizes)
    np_out = np_tile(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out.astype(np.int32)))


def _test_tile_same_dim_int8(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3), dtype=flow.int8, device=flow.device(device)
    )
    size_tensor = flow.Tensor(np.random.randn(4, 2, 3, 19))
    sizes = size_tensor.size()
    of_out = input.tile(reps=sizes)
    np_out = np_tile(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out.astype(np.int32)))


def _test_tile_less_dim_a_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 4, 1, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (2,)
    of_out = input.tile(reps=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np_tile_grad(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


def _test_tile_less_dim_b_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(3, 2, 5),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (3, 4)
    of_out = input.tile(reps=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np_tile_grad(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


def _test_tile_less_dim_c_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(4, 3, 2, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (2, 3, 4, 4)
    of_out = input.tile(reps=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np_tile_grad(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


def _test_tile_same_dim_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(1, 2, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    sizes = (1, 2, 3, 1)
    of_out = input.tile(reps=sizes)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np_tile_grad(input.numpy(), sizes)
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@flow.unittest.skip_unless_1n1d()
class TestTile(flow.unittest.TestCase):
    def test_tile(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_tile_less_dim_a,
            _test_tile_less_dim_b,
            _test_tile_less_dim_c,
            _test_tile_same_dim,
            _test_tile_same_dim_int,
            _test_tile_same_dim_int8,
            _test_tile_less_dim_a_backward,
            _test_tile_less_dim_b_backward,
            _test_tile_less_dim_c_backward,
            _test_tile_same_dim_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
