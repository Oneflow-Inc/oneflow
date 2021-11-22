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


def _count(shape, begin_axis, end_axis):
    cnt = 1
    for i in range(begin_axis, end_axis):
        cnt *= shape[i]
    return cnt


def _l2_norm_numpy(x, dim, epsilon=1e-12):
    square_x_sum_shape = list(x.shape)
    square_x_sum_shape[dim] = 1

    c = x.shape[dim]
    n = int(x.size / c)
    d = _count(x.shape, dim + 1, len(x.shape))

    square_x_sum = np.zeros(square_x_sum_shape)

    square_x_sum_flatten = square_x_sum.reshape(-1)
    in_flatten = x.reshape(-1)
    out = np.zeros(x.size)

    for i in range(0, n):
        offset = int(int((i / d)) * d * c + (i % d))
        for j in range(0, c):
            item = in_flatten[offset + j * d]
            square_x_sum_flatten[i] = square_x_sum_flatten[i] + item * item

        norm = np.sqrt(np.maximum(square_x_sum_flatten[i], epsilon))
        for j in range(0, c):
            index = offset + j * d
            out[index] = in_flatten[index] / norm

    square_x_sum = square_x_sum_flatten.reshape(square_x_sum.shape)
    out = out.reshape(x.shape)
    return out, square_x_sum


def _l2_norm_backward_np(dy, y, square_x_sum, dim, epsilon=1e-12):
    c = dy.shape[dim]
    n = int(dy.size / c)
    d = _count(dy.shape, dim + 1, len(y.shape))

    dx = np.zeros(dy.shape).reshape(-1)
    dy_flatten = dy.reshape(-1)
    y_flatten = y.reshape(-1)
    square_x_sum_flatten = square_x_sum.reshape(-1)

    for i in range(0, n):
        norm = np.sqrt(np.maximum(square_x_sum_flatten[i], epsilon))
        offset = int(int(int((i / d)) * d * c) + (i % d))
        if square_x_sum_flatten[i] >= epsilon:
            y_dy_inner_prod = 0
            for j in range(0, c):
                index = offset + j * d
                y_dy_inner_prod = y_dy_inner_prod + dy_flatten[index] * y_flatten[index]
            for j in range(0, c):
                index = offset + j * d
                dx[index] = (1 / norm) * (
                    dy_flatten[index] - y_dy_inner_prod * y_flatten[index]
                )
        else:
            for j in range(0, c):
                index = offset + j * d
                dx[index] = (1 / norm) * dy_flatten[index]

    return dx.reshape(y.shape)


def _test_l2_normalize(test_case, device, dim, shape):
    input = np.random.randn(*shape)
    np_out, square_x_sum = _l2_norm_numpy(input, dim)
    of_input = flow.tensor(
        input, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    of_out = flow.nn.functional.l2_normalize(of_input, dim)

    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    z = of_out.sum()
    z.backward()
    dx = _l2_norm_backward_np(np.ones(np_out.shape), np_out, square_x_sum, dim)
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), dx, 1e-4, 1e-4))


@flow.unittest.skip_unless_1n1d()
class TestL2Normalize(flow.unittest.TestCase):
    def test_l2_normalize(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_l2_normalize,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dim"] = [0, 1, 2, 3]
        arg_dict["shape"] = [
            (10, 10, 20, 30),
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
