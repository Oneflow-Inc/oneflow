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


@autotest(n=1, check_graph=True, rtol=1e-2, atol=1e-3)
def _test_deconv2d_impl(test_case, placement, input_sbp):
    ndim = 4
    in_channels = random(1, 5).to(int).value() * 8
    groups = random(1, 4).to(int).value()
    out_channels = groups * 8
    kernel_size = random(1, 4).to(int).value()
    stride = random(1, 5).to(int).value()
    padding = random(1, 3).to(int).value()
    dilation = random(1, 5).to(int).value()
    padding_mode = constant("zeros")
    m = torch.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        bias=False,
    )
    m.train(random())

    weight_sbp = random_sbp(placement, max_dim=2, except_partial_sum=True)
    m.weight = torch.nn.Parameter(
        m.weight.to_global(placement=placement, sbp=weight_sbp)
    )

    if m.bias is not None:
        bias_sbp = random_sbp(placement, max_dim=1)
        m.bias = torch.nn.Parameter(m.bias.to_global(placement=placement, sbp=bias_sbp))

    batch = random(1, 3).to(int).value() * 8
    height = random(1, 5).to(int).value() * 8
    width = random(1, 5).to(int).value() * 8
    nchw = [batch, in_channels, height, width]
    x = random_tensor(ndim, *nchw).to_global(placement=placement, sbp=input_sbp)
    y = m(x)
    return y


class TestDeconv2dGlobal(flow.unittest.TestCase):
    @globaltest
    def test_deconv2d(test_case):
        for placement in all_placement():
            for input_sbp in all_sbp(placement, max_dim=2):
                _test_deconv2d_impl(test_case, placement, input_sbp)


if __name__ == "__main__":
    unittest.main()
