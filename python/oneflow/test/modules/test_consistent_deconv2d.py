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

@autotest(n=1, auto_backward=True, check_graph=False)
def _test_deconv2d_impl(test_case, placement, sbp):
    ndim = 4
    channels = random(1, 6).to(int).value() * 8
    m = torch.nn.ConvTranspose2d(
        in_channels=channels,
        out_channels=random(1, 8).to(int).value() * 8,
        kernel_size=random(1, 4).to(int).value(),
        stride=random() | nothing(),
        padding=random(1, 3).to(int).value() | nothing(),
        dilation=random(1, 5).to(int).value() | nothing(),
        groups=random(1, 5).to(int).value() | nothing(),
        padding_mode=constant("zeros") | nothing(),
    )
    m.train(random())
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y

class TestDeconv2dConsistent(flow.unittest.TestCase):
    @globaltest
    def _test_deconv2d_impl(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_deconv2d_impl(test_case, placement, sbp)

if __name__ == "__main__":
    unittest.main()
