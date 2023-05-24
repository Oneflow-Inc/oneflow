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

from random import shuffle
import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


def _test_cast_float2int(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.float32)
    input = flow.tensor(np_arr, dtype=flow.float32, device=flow.device(device))
    output = flow.cast(input, flow.int8)
    np_out = np_arr.astype(np.int8)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_cast_int2float(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.int8)
    input = flow.tensor(np_arr, dtype=flow.int8, device=flow.device(device))
    output = flow.cast(input, flow.float32)
    np_out = np_arr.astype(np.float32)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_cast_bool2int16(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.float32)
    input = flow.tensor(np_arr, dtype=flow.bool, device=flow.device(device))
    output = flow.cast(input, flow.int16)
    np_out = np_arr.astype(bool).astype(np.int16)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_cast_with_non_contiguous_input(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.int8)
    permute_dims = np.arange(len(shape)).tolist()
    shuffle(permute_dims)
    input = flow.tensor(np_arr, dtype=flow.int8, device=flow.device(device)).permute(
        permute_dims
    )
    output = flow.cast(input, flow.float32)
    np_out = np_arr.astype(np.float32).transpose(permute_dims)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))
    test_case.assertTrue(input.stride() == output.stride())


def _test_cast_backward(test_case, device, shape):
    np_arr = np.random.randn(*shape).astype(np.float32)
    x = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = flow.cast(x, flow.float64)
    z = y.sum()
    z.backward()
    np_out = np_arr.astype(np.float64)
    test_case.assertTrue(np.array_equal(x.grad.numpy(), np.ones(shape=shape)))


def random_expand(x, ndim, expand_size):
    dim_size = [1,] * ndim
    random_index = random(0, ndim).to(int).value()
    dim_size[random_index] = expand_size
    return x.expand(*dim_size)


@flow.unittest.skip_unless_1n1d()
class TestCast(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_cast_float2int,
            _test_cast_int2float,
            _test_cast_bool2int16,
            _test_cast_backward,
            # _test_cast_with_non_contiguous_input,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_cast_with_0_size_data(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_cast_float2int,
            _test_cast_int2float,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3, 0, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_cast_with_strided_input(test_case):
        device = random_device()
        x = random_tensor()
        x = x.to(dtype=torch.float32, device=device)
        perm_list = [0, 1, 2, 3]
        shuffle(perm_list)
        x = x.permute(perm_list)
        y = x.to(dtype=torch.float64, device=device)
        return y

    @autotest(n=5)
    def test_cast_with_expanded_input(test_case):
        device = random_device()
        random_expand_size = random(1, 6).to(int).value()
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1)
        x = x.to(dtype=torch.float32, device=device)
        perm_list = [0, 1, 2, 3, 4]
        shuffle(perm_list)
        x = x.permute(perm_list)
        y = random_expand(x, ndim=5, expand_size=random_expand_size)
        z = y.to(dtype=torch.float64, device=device)
        return z

    @autotest(n=5)
    def test_cast_with_expanded_input_2(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=5)
        a = x.to(dtype=torch.float32, device=device)
        b = a.expand((4, 5))
        c = b.to(dtype=torch.double, device=device)
        return c

    @autotest(n=5)
    def test_cast_with_squeezed_input(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.squeeze(x, random(1, 3).to(int))
        z = y.to(dtype=torch.double, device=device)
        return z

    @autotest(n=5, auto_backward=False)
    def test_cast_with_sliced_input(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=20)
        y = random_tensor(ndim=1, dim0=7)
        x = x.to(dtype=torch.float32, device=device)
        y = y.to(device=device)
        rows = x * 10
        cols = y
        a = rows.reshape(20, 1) + cols
        b = a[:, :1]
        c = b.to(torch.int)
        return c

    @autotest(n=5, auto_backward=False)
    # NOTE:if set auto_backward=True, both oneflow and pytorch will raise RuntimeError:
    # element 0 of tensors does not require grad and does not have a grad_fn
    def test_cast_with_scalar_input(test_case):
        device = random_device()
        x = torch.tensor(3.14, device=device)
        y = x.to(dtype=torch.float64, device=device)
        z = y.to(dtype=torch.int8, device=device)
        return z


if __name__ == "__main__":
    unittest.main()
