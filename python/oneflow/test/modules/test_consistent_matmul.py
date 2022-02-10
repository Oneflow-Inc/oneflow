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
def _test_matmul(test_case, placement, x_sbp, y_sbp):
    m = random().to(int).value() * 8
    k = random().to(int).value() * 8
    n = random().to(int).value() * 8
    x = random_tensor(ndim=2, dim0=m, dim1=k).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(ndim=2, dim0=k, dim1=n).to_global(
        placement=placement, sbp=y_sbp
    )
    return torch.matmul(x, y)


@autotest(check_graph=False)
def _test_tensor_matmul(test_case, placement, x_sbp, y_sbp):
    m = random().to(int).value() * 8
    k = random().to(int).value() * 8
    n = random().to(int).value() * 8
    x = random_tensor(ndim=2, dim0=m, dim1=k).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(ndim=2, dim0=k, dim1=n).to_global(
        placement=placement, sbp=y_sbp
    )
    return x.matmul(y)


@autotest(check_graph=False)
def _test_tensor_broadcast_matmul(test_case, placement, x_sbp, y_sbp):
    dim0 = random().to(int).value() * 8
    dim1 = random().to(int).value() * 8
    m = random().to(int).value() * 8
    k = random().to(int).value() * 8
    n = random().to(int).value() * 8
    x = random_tensor(ndim=4, dim0=dim0, dim1=dim1, dim3=k).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(ndim=2, dim0=k, dim1=n).to_global(
        placement=placement, sbp=y_sbp
    )
    return x.matmul(y)


class TestMatMulModule(flow.unittest.TestCase):
    @global_view
    def test_matmul(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    _test_matmul(test_case, placement, x_sbp, y_sbp)
                    _test_tensor_matmul(test_case, placement, x_sbp, y_sbp)
                    _test_tensor_broadcast_matmul(test_case, placement, x_sbp, y_sbp)


if __name__ == "__main__":
    unittest.main()
