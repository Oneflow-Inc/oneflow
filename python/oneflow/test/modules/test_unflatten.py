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


def _test_unflatten(test_case, device):
    m = flow.nn.Unflatten(1, (2, 5, 5))
    x = flow.Tensor(32, 50, 10, device=flow.device(device))
    flow.nn.init.uniform_(x)
    y = m(x)
    test_case.assertTrue(y.shape == flow.Size((32, 2, 5, 5, 10)))
    y2 = flow.unflatten(x, -1, (2, 5))
    test_case.assertTrue(y2.shape == flow.Size((32, 50, 2, 5)))
    y3 = flow.unflatten(x, 0, (1, 32))
    test_case.assertTrue(y3.shape == flow.Size((1, 32, 50, 10)))
    y4 = flow.unflatten(x, 0, (1, 1, 2, 2, 8))
    test_case.assertTrue(y4.shape == flow.Size((1, 1, 2, 2, 8, 50, 10)))


def _test_unflatten_backward(test_case, device):
    m = flow.nn.Unflatten(2, (2, 2)).to(flow.device(device))
    x = flow.Tensor(2, 3, 4, 5, device=flow.device(device))
    x.requires_grad = True
    flow.nn.init.uniform_(x)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.array_equal(np.ones(shape=(2, 3, 4, 5)), x.grad.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestUnflattenModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_unflatten, _test_unflatten_backward]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(True, "Pytorch 1.10.0 do not have unflatten module")
    @profile(torch.unflatten)
    def profile_unflatten(test_case):
        torch.unflatten(torch.ones(1000, 1000), 1, (10, 10, 10))


if __name__ == "__main__":
    unittest.main()
