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
from test_util import GenArgList
from oneflow.test_utils.automated_test_util import *
import numpy as np
import oneflow as flow
import oneflow.unittest


def _count(shape, begin_axis, end_axis):
    cnt = 1
    for i in range(begin_axis, end_axis):
        cnt *= shape[i]
    return cnt


def _l2_norm_numpy(x, dim, epsilon=1e-12):
    axes = [k for k in range(len(list(x.shape)))]
    axes[0], axes[dim] = axes[dim], axes[0]
    axes_tuple = tuple(axes)

    x = np.transpose(x, axes_tuple)

    square_x_sum_shape = list(x.shape)
    square_x_sum_shape[0] = 1

    c = x.shape[0]
    n = int(x.size / c)
    d = _count(x.shape, 1, len(x.shape))

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
    return np.transpose(out, axes_tuple), np.transpose(square_x_sum, axes_tuple)


def _l2_norm_backward_np(dy, y, square_x_sum, dim, epsilon=1e-12):
    axes = [k for k in range(len(list(y.shape)))]
    axes[0], axes[dim] = axes[dim], axes[0]
    axes_tuple = tuple(axes)

    dy = np.transpose(dy, axes_tuple)
    y = np.transpose(y, axes_tuple)
    square_x_sum = np.transpose(square_x_sum, axes_tuple)

    c = dy.shape[0]
    n = int(dy.size / c)
    d = _count(dy.shape, 1, len(y.shape))

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

    return np.transpose(dx.reshape(y.shape), axes_tuple)


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


@flow.unittest.skip_unless_1n1d()
class TestFunctionalNormalize(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_functional_normalize(test_case):
        device = random_device()
        ndim = random(low=2)

        shape = list(random_tensor(ndim).value().shape)
        dim = random(low=0, high=ndim).to(int).value()
        shape[dim] = random(low=2, high=8).to(int).value()
        shape = tuple(shape)

        x = random_pytorch_tensor(len(shape), *shape).to(device)
        y = torch.nn.functional.normalize(x, oneof(2, 3, 4), dim, 1e-12)

        return y


if __name__ == "__main__":
    unittest.main()
