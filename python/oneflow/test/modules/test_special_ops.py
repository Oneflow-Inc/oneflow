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

import torch as torch_original
from packaging import version

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestSpecialOps(flow.unittest.TestCase):
    @autotest(n=5, auto_backward="auto")
    def test_flow_erf_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.erf(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_erfc_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.erfc(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_erfinv_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["float"])
        x = random_tensor(requires_grad=False).to(device).to(x_dtype)
        y = torch.special.erfinv(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_exp2_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.exp2(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_expm1_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.expm1(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_round_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.round(x)

    @autotest(n=5, auto_backward="auto")
    def test_flow_log1p_with_random_data(test_case):
        device = random_device()
        x_dtype = random_dtype(["arithmetic"])
        x = random_tensor().to(device).to(x_dtype)
        y = torch.special.log1p(x)
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_log_softmax_with_random_data(test_case):
        num_dims = random(low=1, high=5).to(int)
        device = random_device()
        x = random_tensor(ndim=num_dims).to(device)
        y = torch.special.log_softmax(x, dim=random(low=0, high=num_dims).to(int))
        return y

    @unittest.skipIf(
        version.parse(torch_original.__version__) <= version.parse("1.13.0"),
        "module 'torch.special' has no attribute 'softmax' before '1.13.0'",
    )
    @autotest(n=5, auto_backward="auto")
    def test_flow_softmax_with_random_data(test_case):
        num_dims = random(low=1, high=5).to(int)
        device = random_device()
        x = random_tensor(ndim=num_dims).to(device)
        y = torch.special.softmax(x, dim=random(low=0, high=num_dims).to(int))
        return y

    @autotest(n=5, auto_backward="auto")
    def test_flow_logsumexp_with_random_data(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        y = torch.special.logsumexp(x, dim=np.random.randint(0, 3))
        return y


if __name__ == "__main__":
    unittest.main()
