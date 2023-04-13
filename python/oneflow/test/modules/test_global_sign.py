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
def _test_sign_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = torch.sign(x)
    return y


class TestSign(flow.unittest.TestCase):
    @globaltest
    def test_sign(test_case):
        for placement in all_placement():
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_sign_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
