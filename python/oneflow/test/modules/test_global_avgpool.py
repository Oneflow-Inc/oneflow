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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_avgpool1d_with_random_data(test_case, placement, sbp):
    m = torch.nn.AvgPool1d(
        kernel_size=random(4, 6),
        stride=random(1, 3),
        padding=random(1, 3),
        ceil_mode=random(),
        count_include_pad=random(),
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 3
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=True)
def _test_avgpool2d_with_random_data(test_case, placement, sbp):
    m = torch.nn.AvgPool2d(
        kernel_size=random(4, 6),
        stride=random(1, 3),
        padding=random(1, 3),
        ceil_mode=random(),
        count_include_pad=random(),
        divisor_override=random().to(int),
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 4
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=True)
def _test_avgpool3d_with_random_data(test_case, placement, sbp):
    m = torch.nn.AvgPool3d(
        kernel_size=random(4, 6),
        stride=random(1, 3),
        padding=random(1, 3),
        ceil_mode=random(),
        count_include_pad=random(),
        divisor_override=random().to(int),
    )
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    ndim = 5
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=True)
def _test_functional_avgpool1d_with_random_data(test_case, placement, sbp):
    ndim = 3
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.avg_pool1d(
        x,
        kernel_size=random(1, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        ceil_mode=random_bool(),
        count_include_pad=random_bool(),
    )
    return y


@autotest(n=1, check_graph=True)
def _test_functional_avgpool2d_with_random_data(test_case, placement, sbp):
    ndim = 4
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.avg_pool2d(
        x,
        kernel_size=random(1, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        ceil_mode=random_bool(),
        count_include_pad=random_bool(),
    )
    return y


@autotest(n=1, check_graph=True)
def _test_functional_avgpool3d_with_random_data(test_case, placement, sbp):
    ndim = 5
    dims = [random(1, 3) * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.avg_pool3d(
        x,
        kernel_size=random(1, 6).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        ceil_mode=random_bool(),
        count_include_pad=random_bool(),
    )
    return y


class TestAvgPoolingModule(flow.unittest.TestCase):
    @globaltest
    def test_avg_pooling(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_avgpool1d_with_random_data(test_case, placement, sbp)
                _test_functional_avgpool1d_with_random_data(test_case, placement, sbp)
            for sbp in all_sbp(placement, max_dim=2):
                _test_avgpool2d_with_random_data(test_case, placement, sbp)
                _test_functional_avgpool2d_with_random_data(test_case, placement, sbp)
            for sbp in all_sbp(placement, max_dim=2):
                _test_avgpool3d_with_random_data(test_case, placement, sbp)
                _test_functional_avgpool3d_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
