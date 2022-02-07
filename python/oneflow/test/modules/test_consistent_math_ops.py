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


@autotest(check_graph=False)
def _test_sinh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.sinh(x)
    return y


@autotest(check_graph=False)
def _test_sin(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.sin(x)
    return y


@autotest(check_graph=False)
def _test_inplace_sin(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = x + 1
    y.sin_()
    return y


@autotest(check_graph=False)
def _test_cos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.cos(x)
    return y


@autotest(check_graph=False)
def _test_log(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.log(x)
    return y


@autotest(check_graph=False)
def _test_sqrt(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.sqrt(x)
    return y


@autotest(check_graph=False)
def _test_exp(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.exp(x)
    return y


@autotest(check_graph=False)
def _test_rsqrt(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.rsqrt(x)
    return y


@autotest(check_graph=False)
def _test_square(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = torch.square(x)
    return y


@autotest(check_graph=False)
def _test_pow_with_scalar(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = random().to(float)
    z = torch.pow(x, y)
    return z


@autotest(auto_backward=False, check_graph=False)
def _test_floordiv_with_scalar(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = random().to(float)
    z = torch.floor_divide(x, y)
    return z


@autotest(check_graph=False)
def _test_arccos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_consistent(
        placement, sbp
    )
    y = torch.arccos(x)
    return y


@autotest(check_graph=False)
def _test_acos(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_consistent(
        placement, sbp
    )
    y = torch.acos(x)
    return y


@autotest(check_graph=False)
def _test_arccosh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_consistent(
        placement, sbp
    )
    y = torch.arccosh(x)
    return y


@autotest(check_graph=False)
def _test_acosh(test_case, placement, sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8, low=2, high=3).to_consistent(
        placement, sbp
    )
    y = torch.acosh(x)
    return y


@autotest(check_graph=False)
def _test_atan2(test_case, placement, x_sbp, y_sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, x_sbp)
    y = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, y_sbp)
    z = torch.atan2(x, y)
    return z


@autotest(check_graph=False)
def _test_minimum(test_case, placement, x_sbp, y_sbp):
    ndim = random(2, 5).to(int).value()
    shape = [8, 8] + [random().to(int).value() for _ in range(ndim - 2)]
    x = random_tensor(ndim, *shape).to_consistent(placement, x_sbp)
    y = random_tensor(ndim, *shape).to_consistent(placement, y_sbp)
    z = torch.minimum(x, y)
    return z


@autotest(check_graph=False)
def _test_broadcast_minimum(test_case, placement, x_sbp, y_sbp):
    k1 = random(2, 6)
    k2 = random(2, 6)
    k3 = random(2, 6)
    x = random_tensor(ndim=5, dim0=8, dim1=8, dim2=k1, dim3=1, dim4=1).to_consistent(
        placement, x_sbp
    )
    y = random_tensor(ndim=5, dim0=8, dim1=8, dim2=1, dim3=k2, dim4=k3).to_consistent(
        placement, y_sbp
    )
    z = torch.minimum(x, y)
    return z


def _test_maximum(test_case, placement, x_sbp, y_sbp):
    ndim = random(2, 5).to(int).value()
    shape = [8, 8] + [random().to(int).value() for _ in range(ndim - 2)]
    x = random_tensor(ndim, *shape).to_consistent(placement, x_sbp)
    y = random_tensor(ndim, *shape).to_consistent(placement, y_sbp)
    z = torch.maximum(x, y)
    return z


@autotest(check_graph=False)
def _test_broadcast_maximum(test_case, placement, x_sbp, y_sbp):
    k1 = random(2, 6)
    k2 = random(2, 6)
    k3 = random(2, 6)
    x = random_tensor(ndim=5, dim0=8, dim1=8, dim2=k1, dim3=1, dim4=1).to_consistent(
        placement, x_sbp
    )
    y = random_tensor(ndim=5, dim0=8, dim1=8, dim2=1, dim3=k2, dim4=k3).to_consistent(
        placement, y_sbp
    )
    z = torch.maximum(x, y)
    return z


@autotest(auto_backward=False, check_graph=False)
def _test_floordiv(test_case, placement, x_sbp, y_sbp):
    ndim = random(2, 5).to(int).value()
    x = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, x_sbp)
    y = random_tensor(ndim=ndim, dim0=8, dim1=8).to_consistent(placement, y_sbp)
    z = torch.floor_divide(x, y)
    return z


class TestMathOps(flow.unittest.TestCase):
    @consistent
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

    @consistent
    def test_binary_api(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    # TODO(): floordiv reports bug when infer shape.
                    # _test_floordiv(test_case, placement, x_sbp, y_sbp)
                    # TODO(): broadcast maximum reports wrong result.
                    # _test_broadcast_maximum(test_case, placement, x_sbp, y_sbp)
                    _test_maximum(test_case, placement, x_sbp, y_sbp)
                    # TODO(): broadcast minimum reports wrong result.
                    # _test_broadcast_minimum(test_case, placement, x_sbp, y_sbp)
                    _test_minimum(test_case, placement, x_sbp, y_sbp)
                    # TODO(): atan2 reports bug when infer shape.
                    # _test_atan2(test_case, placement, x_sbp, y_sbp)


if __name__ == "__main__":
    unittest.main()
