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


@flow.unittest.skip_unless_1n1d()
class TestLogSumExpModule(flow.unittest.TestCase):
    @autotest(n=3, check_graph=True)
    def test_log_sum_exp_against_pytorch(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        y = torch.logsumexp(x, dim=np.random.randint(0, 3))
        return y

    @unittest.skipIf(True, "pytorch-1.10.0 dose not support big_value of logsumexp")
    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_log_sum_exp_with_big_value(test_case):
        device = random_device()
        x = torch.tensor([100, 200]).to(device)
        y = torch.logsumexp(x, dim=0)
        return y

    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_log_sum_exp_with_0_size_tensor(test_case):
        device = random_device()
        x = random_tensor(4, 4, 3, 0, 2).to(device)
        y = torch.logsumexp(x, dim=np.random.randint(0, 3))
        return y

    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_log_sum_exp_with_0dim_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.logsumexp(x, dim=0)
        return y

    @autotest(n=3, check_graph=True)
    def test_tensor_log_sum_exp_against_pytorch(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        y = x.logsumexp(dim=np.random.randint(0, 3))
        return y

    @unittest.skipIf(True, "pytorch-1.10.0 dose not support big_value of logsumexp")
    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_tensor_log_sum_exp_with_big_value(test_case):
        device = random_device()
        x = torch.tensor([100, 200]).to(device)
        y = x.logsumexp(dim=0)
        return y

    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_tensor_log_sum_exp_with_0_size_tensor(test_case):
        device = random_device()
        x = random_tensor(4, 4, 3, 0, 2).to(device)
        y = x.logsumexp(dim=np.random.randint(0, 3))
        return y

    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_tensor_log_sum_exp_with_0dim_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.logsumexp(dim=0)
        return y


if __name__ == "__main__":
    unittest.main()
