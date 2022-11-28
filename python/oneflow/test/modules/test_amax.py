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
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import numpy as np


def __check(test_case, input, dim, keepdim, device):
    of_out = flow.amax(input, dim=dim, keepdim=keepdim)
    if type(dim) is tuple:
        if len(dim) == 0:
            dim = None
    np_out = np.amax(input.numpy(), axis=dim, keepdims=keepdim)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=0.0001, atol=1e-05,))


def _test_amax_with_negative_dim(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    dim = random(-4, 0).to(int).value()
    keepdim = random_bool().value()
    __check(test_case, input, dim, keepdim, device)


def _test_amax_with_positive_dim(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    dim = random(0, 4).to(int).value()
    keepdim = random_bool().value()
    __check(test_case, input, dim, keepdim, device)


def _test_amax_with_multiple_axes(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    axes = set()
    num_axes = random(1, 4).to(int).value()
    for _ in range(num_axes):
        axes.add(random(0, 4).to(int).value())
    keepdim = random_bool().value()
    __check(test_case, input, tuple(axes), keepdim, device)


def _test_amax_with_empty_dim(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    keepdim = random_bool().value()
    __check(test_case, input, None, keepdim, device)


def _test_amax_keepdim(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    dim = random(-4, 4).to(int).value()
    keepdim = True
    __check(test_case, input, dim, keepdim, device)


def _test_amax_not_keepdim(test_case, device):
    input = flow.tensor(
        np.random.randn(3, 5, 6, 8), dtype=flow.float32, device=flow.device(device)
    )
    dim = random(-4, 4).to(int).value()
    keepdim = False
    __check(test_case, input, dim, keepdim, device)


@flow.unittest.skip_unless_1n1d()
class TestAmax(flow.unittest.TestCase):
    def test_amax(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_amax_with_negative_dim,
            _test_amax_with_positive_dim,
            _test_amax_with_multiple_axes,
            _test_amax_with_empty_dim,
            _test_amax_keepdim,
            _test_amax_not_keepdim,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_amax_with_random_data_single_dim(test_case):
        device = random_device()
        ndim = random(1, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        y = torch.amax(x, dim=random(0, ndim), keepdim=random().to(bool))
        return y

    @autotest(n=5)
    def test_amax_with_random_data_empty_dim(test_case):
        device = random_device()
        ndim = random(1, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        y = torch.amax(x, dim=None, keepdim=random().to(bool))
        return y

    @autotest(n=5)
    def test_amax_with_random_data_multi_dims(test_case):
        device = random_device()
        ndim = random(2, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        dim = set()
        for _ in range(random(1, ndim).to(int).value()):
            dim.add(random(0, ndim).to(int).value())
        y = torch.amax(x, dim=tuple(dim), keepdim=random().to(bool))
        return y

    @profile(torch.amax)
    def profile_amax(test_case):
        input1 = torch.ones(4, 4)
        input2 = torch.ones(100, 100)
        torch.amax(input1, 1)
        torch.amax(input1, 1, True)
        torch.amax(input2, 1)
        torch.amax(input2, 1, True)


if __name__ == "__main__":
    unittest.main()
