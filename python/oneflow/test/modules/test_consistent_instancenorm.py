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
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def _test_instancenorm1d_impl(test_case, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for i in range(3)]
    m = torch.nn.InstanceNorm1d(
        num_features=dims[1].to(int),
        eps=random().to(float),
        momentum=random().to(float),
    )
    m.train(random())
    x = random_tensor(3, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_instancenorm2d_impl(test_case, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(4)]
    m = torch.nn.InstanceNorm2d(
        num_features=dims[1].to(int),
        eps=random().to(float),
        momentum=random().to(float),
    )
    m.train(random())
    x = random_tensor(4, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_instancenorm3d_impl(test_case, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(5)]
    m = torch.nn.InstanceNorm3d(
        num_features=dims[1].to(int),
        eps=random().to(float),
        momentum=random().to(float),
    )
    m.train(random())
    x = random_tensor(5, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


class TestInstanceNormConsistent(flow.unittest.TestCase):
    @globaltest
    def test_instancenorm1d(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_instancenorm1d_impl(test_case, placement, sbp)

    @globaltest
    def test_instancenorm2d(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_instancenorm2d_impl(test_case, placement, sbp)

    @globaltest
    def test_instancenorm3d(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_instancenorm3d_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
