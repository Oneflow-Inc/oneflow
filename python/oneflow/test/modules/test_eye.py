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
from automated_test_util import *
from test_util import GenArgList

import oneflow as flow


def _test_eye_forward(test_case, device, n, m):
    output = flow.eye(n, m, device=device)
    np_out = np.eye(n, m)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_eye_backward(test_case, device, n, m):
    x = flow.eye(n, m, device=device)
    x.requires_grad = True
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.array_equal(x.grad.numpy(), np.ones([n, m])))


@flow.unittest.skip_unless_1n1d()
class TestEye(flow.unittest.TestCase):
    def test_eye(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_eye_forward,
            _test_eye_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["n"] = [4, 3, 2]
        arg_dict["m"] = [4, 3, 2]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest()
    def test_eye_with_random_data(test_case):
        n = random().to(int)
        m = random().to(int)
        x = torch.eye(n=n, m=m)
        device = random_device()
        x.to(device)
        x = random_pytorch_tensor().to(device)
        return x


if __name__ == "__main__":
    unittest.main()
