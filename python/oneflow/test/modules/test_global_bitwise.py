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


@autotest(n=1, auto_backward=False)
def _test_bitwise_ops_with_random_data(test_case, op, placement, sbp):
    x = random_tensor(ndim=1, dim0=8, dtype=int).to_global(placement, sbp)
    y = random_tensor(ndim=1, dim0=8, dtype=int).to_global(placement, sbp)
    out = op(x, y)
    return out


@autotest(n=1, auto_backward=False)
def _test_bitwise_not_with_random_data(test_case, placement, sbp):
    x = random_tensor(ndim=1, dim0=8, dtype=int).to_global(placement, sbp)
    return torch.bitwise_not(x)


class TestBitwiseModule(flow.unittest.TestCase):
    @globaltest
    def test_bitwise_and_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_bitwise_ops_with_random_data(
                    test_case, torch.bitwise_and, placement, sbp
                )
                _test_bitwise_ops_with_random_data(
                    test_case, torch.bitwise_or, placement, sbp
                )
                _test_bitwise_ops_with_random_data(
                    test_case, torch.bitwise_xor, placement, sbp
                )
                _test_bitwise_not_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
