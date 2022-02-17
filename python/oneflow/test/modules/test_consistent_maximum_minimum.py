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
def _test_maximum(test_case, placement, x_sbp, y_sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_global(
        placement, x_sbp
    )
    y = random_tensor(ndim=2, dim0=8, dim1=8).to_global(
        placement, y_sbp
    )
    z = torch.maximum(x, y)
    return z

@autotest(check_graph=False)
def _test_minimum(test_case, placement, x_sbp, y_sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_global(
        placement, x_sbp
    )
    y = random_tensor(ndim=2, dim0=8, dim1=8).to_global(
        placement, y_sbp
    )
    z = torch.minimum(x, y)
    return z

@autotest(check_graph=False)
def _test_broadcast_maximum(test_case, placement, x_sbp, y_sbp):
    k1 = random().to(int).value() * 8
    k2 = random().to(int).value() * 8
    k3 = random().to(int).value() * 8

    x = random_tensor(ndim=5, dim0=8, dim1=8, dim2=k1, dim3=1, dim4=k3).to_global(
        placement, x_sbp
    )
    y = random_tensor(ndim=5, dim0=8, dim1=8, dim2=1, dim3=k2, dim4=1).to_global(
        placement, y_sbp
    )
    z = torch.maximum(x, y)
    return z

@autotest(check_graph=False)
def _test_broadcast_minimum(test_case, placement, x_sbp, y_sbp):
    k1 = random().to(int).value() * 8
    k2 = random().to(int).value() * 8
    k3 = random().to(int).value() * 8

    x = random_tensor(ndim=5, dim0=8, dim1=8, dim2=k1, dim3=1, dim4=k3).to_global(
        placement, x_sbp
    )
    y = random_tensor(ndim=5, dim0=8, dim1=8, dim2=1, dim3=k2, dim4=1).to_global(
        placement, y_sbp
    )
    z = torch.minimum(x, y)
    return z


class TestBinaryMathOps(flow.unittest.TestCase):
    @globaltest
    def test_maximum_minimum(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    _test_maximum(test_case, placement, x_sbp, y_sbp)
                    _test_minimum(test_case, placement, x_sbp, y_sbp)
                    
    @globaltest
    def test_broadcast_maximum_minimum(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, valid_split_axis=[0, 1, 2, 4]):
                for y_sbp in all_sbp(placement, valid_split_axis=[0, 1, 3]):
                    _test_broadcast_maximum(test_case, placement, x_sbp, y_sbp)
                    _test_broadcast_minimum(test_case, placement, x_sbp, y_sbp)


if __name__ == "__main__":
    unittest.main()
