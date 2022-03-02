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
def _test_sinh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.sinh(x)
    return y


@autotest(n=1, check_graph=False)
def _test_sin(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.sin(x)
    return y


@autotest(n=1, check_graph=False)
def _test_inplace_sin(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = x + 1
    y.sin_()
    return y


@autotest(n=1, check_graph=False)
def _test_cos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.cos(x)
    return y


@autotest(n=1, check_graph=False)
def _test_log(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.log(x)
    return y


@autotest(n=1, check_graph=False)
def _test_sqrt(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.sqrt(x)
    return y


@autotest(n=1, check_graph=False)
def _test_exp(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.exp(x)
    return y


@autotest(n=1, check_graph=False)
def _test_rsqrt(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.rsqrt(x)
    return y


@autotest(n=1, check_graph=False)
def _test_square(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = torch.square(x)
    return y


@autotest(n=1, check_graph=False)
def _test_pow_with_scalar(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = random().to(float)
    z = torch.pow(x, y)
    return z


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_floordiv_with_scalar(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_global(placement, sbp)
    y = random().to(float)
    z = torch.floor_divide(x, y)
    return z


@autotest(n=1, check_graph=False)
def _test_arccos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_global(
        placement, sbp
    )
    y = torch.arccos(x)
    return y


@autotest(n=1, check_graph=False)
def _test_acos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_global(
        placement, sbp
    )
    y = torch.acos(x)
    return y


@autotest(n=1, check_graph=False)
def _test_arccosh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_global(
        placement, sbp
    )
    y = torch.arccosh(x)
    return y


@autotest(n=1, check_graph=False)
def _test_acosh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_global(
        placement, sbp
    )
    y = torch.acosh(x)
    return y


class TestUnaryMathOps(flow.unittest.TestCase):
    @globaltest
    def test_unary_api(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_sinh(test_case, placement, sbp)
                _test_sin(test_case, placement, sbp)
                _test_inplace_sin(test_case, placement, sbp)
                _test_cos(test_case, placement, sbp)
                _test_log(test_case, placement, sbp)
                _test_sqrt(test_case, placement, sbp)
                _test_exp(test_case, placement, sbp)
                _test_rsqrt(test_case, placement, sbp)
                _test_square(test_case, placement, sbp)
                _test_pow_with_scalar(test_case, placement, sbp)
                _test_floordiv_with_scalar(test_case, placement, sbp)
                _test_arccos(test_case, placement, sbp)
                _test_acos(test_case, placement, sbp)
                _test_arccosh(test_case, placement, sbp)
                _test_acosh(test_case, placement, sbp)


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_floordiv(test_case, placement, x_sbp, y_sbp):
    ndim = random(3, 6).to(int).value()
    dim_list = [random().to(int).value() for _ in range(ndim - 2)]
    x = random_tensor(ndim, 8, 8, *dim_list).to_global(placement, x_sbp)
    y = random_tensor(ndim, 8, 8, *dim_list).to_global(placement, y_sbp)
    z = torch.floor_divide(x, y)
    return z


@autotest(n=1, check_graph=False)
def _test_atan2(test_case, placement, x_sbp, y_sbp):
    ndim = random(3, 6).to(int).value()
    dim_list = [random().to(int).value() for _ in range(ndim - 2)]
    x = random_tensor(ndim, 8, 8, *dim_list).to_global(placement, x_sbp)
    y = random_tensor(ndim, 8, 8, *dim_list).to_global(placement, y_sbp)
    z = torch.atan2(x, y)
    return z


class TestBinaryMathOps(flow.unittest.TestCase):
    @globaltest
    def test_binary_api(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    _test_floordiv(test_case, placement, x_sbp, y_sbp)
                    _test_atan2(test_case, placement, x_sbp, y_sbp)


if __name__ == "__main__":
    unittest.main()
