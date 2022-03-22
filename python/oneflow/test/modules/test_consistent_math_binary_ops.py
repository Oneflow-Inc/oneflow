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


@autotest(n=1, check_graph=False)
def _test_pow_with_scalar(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    y = random().to(float)
    z = torch.pow(x, y)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_floordiv_with_scalar(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list,).to_global(placement, sbp)
    y = random().to(float)
    z = torch.floor_divide(x, y)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_floordiv(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    y = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    z = torch.floor_divide(x, y)
    return z


@autotest(n=1, check_graph=False)
def _test_atan2(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    y = random_tensor(ndim, *dim_list).to_global(placement, sbp)
    z = torch.atan2(x, y)
    return z


class TestMathOps(flow.unittest.TestCase):
    @globaltest
    def test_math_ops(test_case):
        ndim = random().to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_pow_with_scalar(test_case, placement, sbp, ndim)
                _test_floordiv_with_scalar(test_case, placement, sbp, ndim)
                _test_floordiv(test_case, placement, sbp, ndim)
                _test_atan2(test_case, placement, sbp, ndim)


if __name__ == "__main__":
    unittest.main()
