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
def _test_broadcast_mul(test_case, placement, sbp):
    x = random_tensor(ndim=3, dim0=16, dim1=8, dim2=24).to_global(placement, sbp)
    y_sbp = random_sbp(placement, max_dim=2)
    y = random_tensor(ndim=2, dim0=8, dim1=24).to_global(placement, y_sbp)
    z = torch.mul(x, y)
    return z


@autotest(n=1, check_graph=True)
def _test_mul_with_scalar(test_case, ndim, placement, sbp):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    y = 2
    return torch.mul(x, y)


class TestMulModule(flow.unittest.TestCase):
    @globaltest
    def test_broadcast_mul(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_broadcast_mul(test_case, placement, sbp)

    @globaltest
    def test_mul_with_scalar(test_case):
        ndim = random(1, 4).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_mul_with_scalar(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
