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
from oneflow.test_utils.automated_test_util import *
import oneflow.unittest


@autotest(n=1, check_graph=True)
def _test_pad_1d_impl(test_case, placement, sbp):
    pad = [random(0, 5).to(int) for i in range(2)]
    x = random_tensor(
        ndim=3, dim0=8, dim1=random(2, 8).to(int) * 8, dim2=random(2, 8).to(int) * 8
    ).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.pad(x, pad, mode=oneof("constant", "reflect", "replicate"))
    return y


@autotest(n=1, check_graph=True)
def _test_pad_2d_impl(test_case, placement, sbp):
    pad = [random(0, 5).to(int) for i in range(4)]
    x = random_tensor(
        ndim=4,
        dim0=8,
        dim1=8,
        dim2=random(2, 8).to(int) * 8,
        dim3=random(2, 8).to(int) * 8,
    ).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.pad(x, pad, mode=oneof("constant", "reflect", "replicate"))
    return y


class TestPad(flow.unittest.TestCase):
    @globaltest
    def test_pad_1d(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_pad_1d_impl(test_case, placement, sbp)
                _test_pad_2d_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
