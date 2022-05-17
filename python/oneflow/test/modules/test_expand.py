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

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def _np_get_expand(input_shape, expand_size):
    input = np.random.random(size=input_shape).astype(np.float32)

    input_stride = [1]
    for i in range(len(input_shape) - 2, -1, -1):
        input_stride.insert(0, input_stride[0] * input_shape[i + 1])
    # calculate the output shape and stride
    new_size = []
    new_stride = []
    diff = len(expand_size) - len(input_shape)
    for i in range(len(expand_size) - 1, -1, -1):
        if i >= diff:
            if expand_size[i] == -1 or expand_size[i] == input_shape[i - diff]:
                new_size.insert(0, input_shape[i - diff])
                new_stride.insert(0, input_stride[i - diff])
            else:
                assert expand_size[i] >= 1 and input_shape[i - diff] == 1
                new_size.insert(0, expand_size[i])
                new_stride.insert(0, 0)
        else:
            assert expand_size[i] >= 1
            new_size.insert(0, expand_size[i])
            if expand_size[i] == 1:
                new_stride.insert(0, new_stride[0])
            else:
                new_stride.insert(0, 0)

    gout = np.random.random(size=tuple(new_size)).astype(np.float32)

    out_stride = [1]
    for i in range(len(new_size) - 2, -1, -1):
        out_stride.insert(0, out_stride[0] * new_size[i + 1])

    gin = np.zeros(input_shape).flatten()
    out = np.zeros(np.product(new_size))

    def getOffset(i_offset, stride, expand_stride, n):
        remain = i_offset
        o_offset = 0
        for i in range(n):
            idx = int(remain / stride[i])
            o_offset += idx * expand_stride[i]
            remain = remain - idx * stride[i]
        return o_offset

    in_flatten = input.flatten()
    gout_flatten = gout.flatten()
    num_elem = np.product(new_size)
    dims = len(new_size)

    for i in range(num_elem):
        offset = getOffset(i, out_stride, new_stride, dims)
        gin[offset] += gout_flatten[i]
        out[i] = in_flatten[offset]

    return input, gout, out.reshape(tuple(new_size)), gin.reshape(input_shape)


def _test_expand_new_dims(test_case, device):
    input_shape = (1, 4, 1, 32)
    expand_dim = [2, 1, 2, 4, 2, 32]
    input, gout, out_np, gin_np = _np_get_expand(input_shape, expand_dim)
    of_input = flow.tensor(
        input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input.expand(2, 1, 2, 4, 2, 32)
    test_case.assertTrue(np.array_equal(of_out.numpy(), out_np))


def _test_expand_same_dim(test_case, device):
    input_shape = (2, 4, 1, 32)
    expand_dim = [2, 4, 2, 32]
    input, gout, out_np, gin_np = _np_get_expand(input_shape, expand_dim)
    of_input = flow.tensor(input, dtype=flow.float32, device=flow.device(device))
    of_out = of_input.expand(2, 4, 2, 32)

    test_case.assertTrue(np.array_equal(of_out.numpy(), out_np))


def _test_expand_same_dim_negative(test_case, device):
    input_shape = (1, 6, 5, 3)
    expand_dim = [4, -1, 5, 3]
    input, gout, out_np, gin_np = _np_get_expand(input_shape, expand_dim)
    of_input = flow.tensor(input, dtype=flow.float32, device=flow.device(device))
    of_out = of_input.expand(4, -1, 5, 3)

    test_case.assertTrue(np.array_equal(of_out.numpy(), out_np))


def _test_expand_same_int(test_case, device):
    input_shape = (2, 4, 1, 32)
    expand_dim = [2, 4, 2, 32]
    input, gout, out_np, gin_np = _np_get_expand(input_shape, expand_dim)
    of_input = flow.tensor(input, dtype=flow.int, device=flow.device(device))
    of_out = of_input.expand(2, 4, 2, 32)

    test_case.assertTrue(np.array_equal(of_out.numpy(), out_np.astype(np.int32)))


def _test_expand_flow_size(test_case, device):
    input_shape = (2, 4, 1, 32)
    expand_dim = flow.Size([2, 4, 2, 32])
    input, gout, out_np, gin_np = _np_get_expand(input_shape, expand_dim)
    of_input = flow.tensor(input, dtype=flow.int, device=flow.device(device))
    of_out = of_input.expand(expand_dim)

    test_case.assertTrue(np.array_equal(of_out.numpy(), out_np.astype(np.int32)))


def _test_expand_backward_same_dim(test_case, device):
    input = np.array(
        [
            [
                [[0.9876952171325684]],
                [[0.8772538304328918]],
                [[0.9200366735458374]],
                [[0.2810221314430237]],
            ],
            [
                [[0.3037724494934082]],
                [[0.7783719897270203]],
                [[0.08884672075510025]],
                [[0.17156553268432617]],
            ],
        ]
    )
    of_input = flow.tensor(
        input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input.expand(2, 4, 2, 1)
    of_out.sum().backward()
    np_grad = [
        [[[2.0]], [[2.0]], [[2.0]], [[2.0]]],
        [[[2.0]], [[2.0]], [[2.0]], [[2.0]]],
    ]
    test_case.assertTrue(np.array_equal(of_input.grad.numpy(), np_grad))


def _test_expand_backward(test_case, device):
    input = np.array(
        [
            [
                [[0.8981702327728271, 0.5372866988182068]],
                [[0.45116370916366577, 0.8656941056251526]],
                [[0.8811476230621338, 0.5552017688751221]],
                [[0.6291894316673279, 0.5786571502685547]],
            ]
        ]
    )
    of_input = flow.tensor(
        input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = of_input.expand(2, 1, 2, 4, 2, 2)
    of_out.sum().backward()
    np_grad = [[[[8.0, 8.0]], [[8.0, 8.0]], [[8.0, 8.0]], [[8.0, 8.0]]]]
    test_case.assertTrue(np.array_equal(of_input.grad.numpy(), np_grad))


def random_expand(x, ndim, expand_size):
    dim_size = [1,] * ndim
    random_index = random(0, ndim).to(int).value()
    dim_size[random_index] = expand_size
    return x.expand(*dim_size)


@flow.unittest.skip_unless_1n1d()
class TestExpand(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_tensor_expand_with_random_data(test_case):
        random_expand_size = random(1, 6).to(int).value()
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1)
        return random_expand(x, ndim=5, expand_size=random_expand_size)

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_tensor_expand_bool_with_random_data(test_case):
        random_expand_size = random(1, 6).to(int).value()
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1).to(torch.bool)
        return random_expand(x, ndim=5, expand_size=random_expand_size)

    def test_expand_compare_with_numpy(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_expand_new_dims,
            _test_expand_same_dim,
            _test_expand_same_dim_negative,
            _test_expand_same_int,
            _test_expand_flow_size,
            _test_expand_backward,
            _test_expand_backward_same_dim,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, auto_backward=False)
    def test_flow_expand_with_0_size(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=1).to(device)
        return x.expand([0, 3])


if __name__ == "__main__":
    unittest.main()
