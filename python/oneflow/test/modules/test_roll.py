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

import torch


def _test_roll(test_case, device):
    torch_x = torch.rand(
        (2, 3, 5, 10, 20), device=device, dtype=torch.float32, requires_grad=True
    )
    torch_grad = torch.rand_like(torch_x, device=device)

    shifts = (
        np.random.randint(-100, 100),
        np.random.randint(-100, 100),
        np.random.randint(-100, 100),
        np.random.randint(-100, 100),
    )
    dims = (0, 2, 3, 4)

    torch_y = torch.roll(torch_x, shifts, dims)
    torch_y.backward(torch_grad)

    of_x = flow.tensor(
        torch_x.detach().cpu().numpy(),
        device=device,
        dtype=flow.float32,
        requires_grad=True,
    )
    of_y = flow.roll(of_x, shifts, dims)
    of_grad = flow.tensor(torch_grad.cpu().numpy(), device=device, dtype=flow.float32)
    of_y.backward(of_grad)

    test_case.assertTrue(np.array_equal(of_y.numpy(), torch_y.detach().cpu().numpy()))
    test_case.assertTrue(np.array_equal(of_x.grad.numpy(), torch_x.grad.cpu().numpy()))


def _test_roll_single_dims(test_case, device):
    torch_x = torch.rand(
        (2, 3, 5, 10, 20), device=device, dtype=torch.float32, requires_grad=True
    )
    torch_grad = torch.rand_like(torch_x, device=device)

    shifts = np.random.randint(-100, 100)
    dims = np.random.randint(0, 4)

    torch_y = torch.roll(torch_x, shifts, dims)
    torch_y.backward(torch_grad)

    of_x = flow.tensor(
        torch_x.detach().cpu().numpy(),
        device=device,
        dtype=flow.float32,
        requires_grad=True,
    )
    of_y = flow.roll(of_x, shifts, dims)
    of_grad = flow.tensor(torch_grad.cpu().numpy(), device=device, dtype=flow.float32)
    of_y.backward(of_grad)

    test_case.assertTrue(np.array_equal(of_y.numpy(), torch_y.detach().cpu().numpy()))
    test_case.assertTrue(np.array_equal(of_x.grad.numpy(), torch_x.grad.cpu().numpy()))


def _test_roll_none_dims(test_case, device):
    torch_x = torch.rand(
        (2, 3, 5, 10, 20), device=device, dtype=torch.float32, requires_grad=True
    )
    torch_grad = torch.rand_like(torch_x, device=device)

    shifts = np.random.randint(-100, 100)
    dims = None

    torch_y = torch.roll(torch_x, shifts, dims)
    torch_y.backward(torch_grad)

    of_x = flow.tensor(
        torch_x.detach().cpu().numpy(),
        device=device,
        dtype=flow.float32,
        requires_grad=True,
    )
    of_y = flow.roll(of_x, shifts, dims)
    of_grad = flow.tensor(torch_grad.cpu().numpy(), device=device, dtype=flow.float32)
    of_y.backward(of_grad)

    test_case.assertTrue(np.array_equal(of_y.numpy(), torch_y.detach().cpu().numpy()))
    test_case.assertTrue(np.array_equal(of_x.grad.numpy(), torch_x.grad.cpu().numpy()))


@flow.unittest.skip_unless_1n1d()
class TestRoll(flow.unittest.TestCase):
    def test_expand_compare_with_torch(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_roll,
            _test_roll_single_dims,
            _test_roll_none_dims,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
