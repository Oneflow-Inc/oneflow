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

from oneflow.test_utils.automated_test_util import *
from oneflow.nn.common_types import _size_2_t, _size_4_t, _size_6_t
import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_constantpad1d(test_case, placement, sbp):
    m = torch.nn.ConstantPad1d(
        padding=random(1, 6).to(_size_2_t), value=random().to(float)
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 3
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_constantpad2d(test_case, placement, sbp):
    m = torch.nn.ConstantPad2d(
        padding=random(1, 6).to(_size_4_t), value=random().to(float)
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 4
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_functional_constantpad2d(test_case, placement, sbp):
    padding = random(-1, 6).to(_size_4_t)
    value = random().to(float)
    ndim = 4
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.pad(x, pad=padding, mode="constant", value=value)
    return y


@autotest(n=1, check_graph=False)
def _test_constantpad3d(test_case, placement, sbp):
    m = torch.nn.ConstantPad2d(
        padding=random(1, 6).to(_size_6_t), value=random().to(float)
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 5
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


class TestConstantPad(flow.unittest.TestCase):
    @globaltest
    def test_constantpad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_constantpad1d(test_case, placement, sbp)
                _test_constantpad2d(test_case, placement, sbp)
                _test_constantpad3d(test_case, placement, sbp)
                _test_functional_constantpad2d(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
