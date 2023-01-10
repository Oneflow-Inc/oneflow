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


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_cumsum_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    dim = random(0, ndim).to(int).value()
    z = torch.cumsum(x, dim)
    return z


@unittest.skip("This fails in multi-gpu")
class TestCumsumGlobal(flow.unittest.TestCase):
    @globaltest
    def test_cumsum(test_case):
        # random ndim in range [1,4]
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_cumsum_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
