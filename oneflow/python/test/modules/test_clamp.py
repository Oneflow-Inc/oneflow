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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_clamp(test_case, shape, device):
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.clamp(input, 0.1, 0.5)
    np_out = np.clip(input.numpy(), 0.1, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_tensor_clamp(test_case, shape, device):
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = input.clamp(0.1, 0.5)
    np_out = np.clip(input.numpy(), 0.1, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_clamp_scalar_min(test_case, shape, device):
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.clamp(input, 0.1, None)
    np_out = np.clip(input.numpy(), 0.1, None)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_clamp_scalar_max(test_case, shape, device):
    input = flow.Tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.clamp(input, None, 0.5)
    np_out = np.clip(input.numpy(), None, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_clamp_integral(test_case, shape, device):
    input = flow.Tensor(np.random.randint(3, 10, (shape)), device=flow.device(device))
    of_out = flow.clamp(input, 1, 5)
    np_out = np.clip(input.numpy(), 1, 5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _numpy_clamp_grad(arr, min, max):
    grad = np.zeros_like(arr)
    grad[arr.clip(min, max) == arr] += 1
    return grad


def _test_clamp_backward(test_case, shape, device):
    x = flow.Tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.clamp(x, 0.1, 0.5).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), _numpy_clamp_grad(x.numpy(), 0.1, 0.5), 1e-5, 1e-5)
    )


class TestClampModule(flow.unittest.TestCase):
    def test_clamp(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_clamp,
            _test_tensor_clamp,
            _test_clamp_scalar_min,
            _test_clamp_scalar_max,
            _test_clamp_integral,
            _test_clamp_backward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
