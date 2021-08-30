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

import random as rd
import unittest
from collections import OrderedDict

import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


def _numpy_fmod(x, y):
    sign = np.sign(x)
    res = np.fmod(np.abs(x), np.abs(y))
    return sign * res


def _numpy_fmod_grad(x):
    grad = np.ones_like(x)
    return grad


def _test_fmod_same_shape_tensor(test_case, shape, device):
    input = flow.Tensor(
        np.random.uniform(-100, 100, shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    other = flow.Tensor(
        np.random.uniform(-10, 10, shape),
        dtype=flow.float32,
        device=flow.device(device),
    )
    of_out = flow.fmod(input, other)
    np_out = _numpy_fmod(input.numpy(), other.numpy())
    of_out.sum().backward()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), _numpy_fmod_grad(input.numpy()), 1e-05, 1e-05)
    )


def _test_fmod_tensor_vs_scalar(test_case, shape, device):
    input = flow.Tensor(
        np.random.randint(-100, 100, shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    other = rd.uniform(-1, 1) * 100
    of_out = flow.fmod(input, other)
    np_out = _numpy_fmod(input.numpy(), other)
    of_out.sum().backward()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), _numpy_fmod_grad(input.numpy()), 1e-05, 1e-05)
    )


class TestFmodModule(flow.unittest.TestCase):
    @autotest
    def test_flow_fmod_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        other = random_pytorch_tensor().to(device)
        return torch.fmod(input, other)

    @autotest(auto_backward=False)
    def test_flow_tensor_fmod_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.fmod(x, 2)
        return y


if __name__ == "__main__":
    unittest.main()
