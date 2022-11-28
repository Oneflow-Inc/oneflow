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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_mean(test_case, shape, device):
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=1)
    np_out = np.mean(input.numpy(), axis=1)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=0)
    np_out = np.mean(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


def _test_mean_negative_dim(test_case, shape, device):
    if len(shape) < 4:
        shape = (2, 3, 4, 5)
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mean(input, dim=(-2, -1, -3))
    np_out = np.mean(input.numpy(), axis=(-2, -1, -3))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


def _test_mean_backward(test_case, shape, device):
    np_arr = np.random.randn(*shape)
    x = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = flow.mean(x, dim=1)
    z = y.sum()
    z.backward()
    np_grad = np.zeros(shape=np_arr.shape)
    np_grad[:] = 1 / x.size(1)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestMean(flow.unittest.TestCase):
    def test_mean(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_mean,
            _test_mean_negative_dim,
            _test_mean_backward,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True)
    def test_mean_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float).to(device)
        return torch.mean(x, dim)

    @autotest(n=5)
    def test_mean_with_scalar_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float).to(device).mean()
        y = x.mean(-1)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=5, atol=1e-3)
    def test_mean_with_float16_data(test_case):
        device = gpu_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float).to(device=device, dtype=torch.float16)
        return torch.mean(x, dim)


if __name__ == "__main__":
    unittest.main()
