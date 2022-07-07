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
def test_flow_tensor_broadcast_matmul_with_random_data(
    test_case, placement, x_sbp, y_sbp
):
    k = random(1, 6)
    x = random_tensor(ndim=3, dim2=k).to_global(placement=placement, sbp=x_sbp)
    y = random_tensor(ndim=2, dim0=k).to_global(placement=placement, sbp=y_sbp)
    return x.matmul(y)


@autotest(n=1, check_graph=False)
def test_flow_tensor_x_broadcast_y_matmul(test_case, placement, x_sbp, y_sbp):
    k = random(1, 6).to(int)
    x = random_tensor(ndim=2, dim1=k).to_global(placement=placement, sbp=x_sbp)
    y = random_tensor(ndim=3, dim1=k).to_global(placement=placement, sbp=y_sbp)
    return x.matmul(y)


@autotest(n=1, check_graph=False)
def test_flow_tensor_broadcast_matmul_with_same_dims(
    test_case, placement, x_sbp, y_sbp
):
    k = random(1, 6).to(int)
    x = random_tensor(ndim=2, dim0=1, dim1=k).to_global(placement=placement, sbp=x_sbp)
    y = random_tensor(ndim=2, dim0=k, dim1=1).to_global(placement=placement, sbp=y_sbp)
    return x.matmul(y)


class TestBroadcastMatmulModule(flow.unittest.TestCase):
    @globaltest
    def test_broadcast_matmul(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    test_flow_tensor_broadcast_matmul_with_random_data(
                        test_case, placement, x_sbp, y_sbp
                    )
                    test_flow_tensor_x_broadcast_y_matmul(
                        test_case, placement, x_sbp, y_sbp
                    )
                    test_flow_tensor_broadcast_matmul_with_same_dims(
                        test_case, placement, x_sbp, y_sbp
                    )


if __name__ == "__main__":
    unittest.main()
