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

from oneflow.test_utils.automated_test_util import *


def _test_clamp(test_case, shape, device, dtype):
    input = flow.tensor(
        np.random.randn(*shape), dtype=dtype, device=flow.device(device)
    )
    of_out = flow.clamp(input, 0.1, 0.5)
    np_out = np.clip(input.numpy(), 0.1, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_tensor_clamp(test_case, shape, device, dtype):
    input = flow.tensor(
        np.random.randn(*shape), dtype=dtype, device=flow.device(device)
    )
    of_out = input.clamp(0.1, 0.5)
    np_out = np.clip(input.numpy(), 0.1, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_scalar_min(test_case, shape, device, dtype):
    input = flow.tensor(
        np.random.randn(*shape), dtype=dtype, device=flow.device(device)
    )
    of_out = flow.clamp(input, 0.1, None)
    np_out = np.clip(input.numpy(), 0.1, None)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_scalar_max(test_case, shape, device, dtype):
    input = flow.tensor(
        np.random.randn(*shape), dtype=dtype, device=flow.device(device)
    )
    of_out = flow.clamp(input, None, 0.5)
    np_out = np.clip(input.numpy(), None, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_integral(test_case, shape, device, dtype):
    input = flow.tensor(np.random.randint(3, 10, shape), device=flow.device(device)).to(
        dtype
    )
    of_out = flow.clamp(input, 1, 5)
    np_out = np.clip(input.numpy(), 1, 5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _numpy_clamp_grad(arr, min, max):
    grad = np.zeros_like(arr)
    grad[arr.clip(min, max) == arr] += 1
    return grad


def _test_clamp_backward(test_case, shape, device, dtype):
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.clamp(x, 0.1, 0.5).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), _numpy_clamp_grad(x.numpy(), 0.1, 0.5), 1e-05, 1e-05
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestClampModule(flow.unittest.TestCase):
    def test_clamp(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_clamp,
            _test_tensor_clamp,
            _test_clamp_scalar_min,
            _test_clamp_scalar_max,
            _test_clamp_backward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [flow.float16, flow.float, flow.double]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

        arg_dict["fun"] = [
            _test_clamp_integral,
        ]
        arg_dict["dtype"] = [flow.int8, flow.int, flow.long]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_clamp_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clamp(input, min=random().to(float), max=random().to(float))
        return y

    @autotest(n=5)
    def test_clamp_min_none_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clamp(input, min=random().to(float), max=random().to(float))
        return y

    @autotest(n=5)
    def test_clamp_max_none_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clamp(
            input, min=random().to(float), max=random().to(float) | nothing()
        )
        return y

    @profile(torch.clamp)
    def profile_clamp(test_case):
        torch.clamp(torch.ones(4), -1, 2)
        torch.clamp(torch.ones(100000), -1, 2)

    @autotest(n=5)
    def test_clip_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clip(input, min=random().to(float), max=random().to(float))
        return y

    @autotest(n=5)
    def test_clip_min_none_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clip(input, min=random().to(float), max=random().to(float))
        return y

    @autotest(n=5)
    def test_clip_max_none_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clip(
            input, min=random().to(float), max=random().to(float) | nothing()
        )
        return y

    @profile(torch.clip)
    def profile_clip(test_case):
        torch.clip(torch.ones(4), -1, 2)
        torch.clip(torch.ones(100000), -1, 2)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_clamp_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.clamp(x, min=random().to(float), max=random().to(float))
        return y


def _test_clamp_min(test_case, shape, device):
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.clamp_min(input, 0.1)
    np_out = np.clip(input.numpy(), 0.1, None)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_min_integral(test_case, shape, device):
    input = flow.tensor(np.random.randint(3, 10, shape), device=flow.device(device))
    of_out = flow.clamp_min(input, 1)
    np_out = np.clip(input.numpy(), 1, None)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_min_backward(test_case, shape, device):
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.clamp_min(x, 0.1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), _numpy_clamp_grad(x.numpy(), 0.1, None), 1e-05, 1e-05
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestClampMinModule(flow.unittest.TestCase):
    def test_clamp_min(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_clamp_min,
            _test_clamp_min_integral,
            _test_clamp_min_backward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_clamp_min_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clamp_min(input, min=random().to(float))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_clamp_min_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.clamp_min(x, min=random().to(float))
        return y


def _test_clamp_max(test_case, shape, device):
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.clamp_max(input, 0.5)
    np_out = np.clip(input.numpy(), None, 0.5)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_max_integral(test_case, shape, device):
    input = flow.tensor(np.random.randint(3, 10, shape), device=flow.device(device))
    of_out = flow.clamp_max(input, 1)
    np_out = np.clip(input.numpy(), None, 1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_clamp_max_backward(test_case, shape, device):
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.clamp_max(x, 0.5).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(
            x.grad.numpy(), _numpy_clamp_grad(x.numpy(), None, 0.5), 1e-05, 1e-05
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestClampMaxModule(flow.unittest.TestCase):
    def test_clamp_min(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_clamp_max,
            _test_clamp_max_integral,
            _test_clamp_max_backward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_clamp_max_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = torch.clamp_max(input, max=random().to(float))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_clamp_max_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.clamp_max(x, max=random().to(float))
        return y


if __name__ == "__main__":
    unittest.main()
