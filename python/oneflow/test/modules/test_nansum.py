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
class TestNanSumModule(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True)
    def test_nansum_without_nan(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        y = torch.nansum(x)
        return y

    @autotest(n=5, check_graph=True)
    def test_nansum_with_partial_nan(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        mask = x < 0
        x = x.masked_fill(mask, float("nan"))
        y = torch.nansum(x)
        return y

    @autotest(n=5, check_graph=True)
    def test_nansum_with_total_nan(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        mask = torch.ones_like(x).bool()
        x = x.masked_fill(mask, float("nan"))
        y = torch.nansum(x)
        return y

    @autotest(n=5, check_graph=True)
    def test_nansum_with_partial_nan_dims(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        mask = x < 0
        x = x.masked_fill(mask, float("nan"))
        y = torch.nansum(x, dim=random(0, 4).to(int))
        return y

    @autotest(n=5, check_graph=True)
    def test_nansum_with_total_nan_dims(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        mask = torch.ones_like(x).bool()
        x = x.masked_fill(mask, float("nan"))
        y = torch.nansum(x, dim=random(0, 4).to(int))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_sum_with_0_size_tensor(test_case):
        device = random_device()
        x = random_tensor(4, 4, 3, 0, 2).to(device)
        y = torch.nansum(x, dim=np.random.randint(0, 3))
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_sum_with_0dim_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.nansum(x)
        return y


if __name__ == "__main__":
    unittest.main()
