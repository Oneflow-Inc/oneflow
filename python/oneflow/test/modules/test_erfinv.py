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
from scipy import special
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_flow_erfinv_with_inf_data(test_case, device):
    x = flow.tensor(np.ones((5, 5)), dtype=flow.float32, device=flow.device(device))
    of_out = flow.erfinv(x)
    np_out = np.full((5, 5), np.inf)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_flow_erfinv_with_nan_data(test_case, device):
    x = flow.tensor(
        np.arange(2, 22).reshape(4, 5), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.erfinv(x)
    np_out = np.full((4, 5), np.nan)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out, equal_nan=True))


@flow.unittest.skip_unless_1n1d()
class TestErfinvModule(flow.unittest.TestCase):
    def test_flow_erfinv(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_flow_erfinv_with_inf_data,
            _test_flow_erfinv_with_nan_data,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True, auto_backward=False)
    def test_flow_erfinv_with_random_data(test_case):
        device = random_device()
        x = random_tensor(requires_grad=False).to(device)
        y = torch.erfinv(x)
        return y

    @profile(torch.erfinv)
    def profile_erfinv(test_case):
        torch.erfinv(torch.ones(100000))


if __name__ == "__main__":
    unittest.main()
