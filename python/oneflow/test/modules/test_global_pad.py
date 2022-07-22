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


@autotest(n=1, check_graph=False)
def _test_pad_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    pad = []
    for i in range(ndim * 2):
        pad.append(1)
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.pad(x, pad, mode=oneof("constant", "reflect", "replicate"))
    return y


class TestPad(flow.unittest.TestCase):
    @globaltest
    def test_pad(test_case):
        for placement in all_placement():
            ndim = random(2, 6).to(int).value()
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_pad_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
