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

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_conv3d(test_case, placement, sbp):
    ndim = 5
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    max_groups = dims[1].value() // 8
    m = torch.nn.Conv3d(
        in_channels=dims[1],
        out_channels=random(1, 3).to(int) * 8,
        kernel_size=random(1, 4).to(int),
        stride=random(1, 5).to(int),
        padding=random(1, 3).to(int),
        dilation=random(1, 5).to(int),
        groups=random(1, max_groups + 1).to(int),
    )
    m.train(random())
    m.weight = torch.nn.Parameter(m.weight.to_global(placement=placement, sbp=sbp))
    if m.bias is not None:
        m.bias = torch.nn.Parameter(
            m.bias.to_global(placement=placement, sbp=random_sbp(placement, max_dim=1))
        )
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def test_depthwise_conv3d(test_case, placement, sbp):
    ndim = 5
    dims = [random(1, 3).to(int) * 8 for _ in range(ndim)]
    m = torch.nn.Conv3d(
        in_channels=dims[1],
        out_channels=dims[1],
        kernel_size=random(1, 4).to(int),
        stride=random(1, 3).to(int),
        padding=random(1, 3).to(int),
        dilation=random(1, 5).to(int),
        groups=dims[1],
    )
    m.train(random())
    m.weight = torch.nn.Parameter(m.weight.to_global(placement=placement, sbp=sbp))
    if m.bias is not None:
        m.bias = torch.nn.Parameter(
            m.bias.to_global(placement=placement, sbp=random_sbp(placement, max_dim=1))
        )
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)


class TestConv3d(flow.unittest.TestCase):
    @globaltest
    def test_conv3d(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_conv3d(test_case, placement, sbp)
            for sbp in all_sbp(placement, max_dim=1):
                test_depthwise_conv3d(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
