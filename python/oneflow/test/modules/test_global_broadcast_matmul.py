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
def _test_flow_tensor_global_broadcast_matmul_with_random_data(
    test_case, placement, x_sbp, y_sbp
):
    batch_dim = random(1, 6) * 8
    k = random(1, 6) * 4
    x = random_tensor(ndim=3, dim0=batch_dim, dim2=k).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(ndim=2, dim0=k).to_global(placement=placement, sbp=y_sbp)
    return x.matmul(y)


@autotest(n=1, check_graph=True)
def _test_flow_tensor_global_x_broadcast_y_matmul(test_case, placement, x_sbp, y_sbp):
    batch_dim = random(1, 6) * 8
    k = random(1, 6) * 4
    x = random_tensor(ndim=2, dim1=k).to_global(placement=placement, sbp=x_sbp)
    y = random_tensor(ndim=3, dim0=batch_dim, dim1=k).to_global(
        placement=placement, sbp=y_sbp
    )

    return x.matmul(y)


@autotest(n=1, check_graph=True)
def _test_flow_tensor_global_broadcast_matmul_with_same_dims(
    test_case, placement, x_sbp, y_sbp
):
    k = random(1, 6) * 8
    batch_dim = random(1, 6) * 8
    x = random_tensor(ndim=3, dim0=batch_dim, dim1=4, dim2=k).to_global(
        placement=placement, sbp=x_sbp
    )
    y = random_tensor(ndim=3, dim0=batch_dim, dim1=k, dim2=4).to_global(
        placement=placement, sbp=y_sbp
    )
    return x.matmul(y)


class TestGlobalBroadcastMatmulModule(flow.unittest.TestCase):
    @globaltest
    def test_global_broadcast_matmul_with_random_data(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2, valid_split_axis=[0]):
                for y_sbp in all_sbp(placement, max_dim=2, except_split=True):
                    _test_flow_tensor_global_broadcast_matmul_with_random_data(
                        test_case, placement, x_sbp, y_sbp
                    )

    @globaltest
    def test_global_x_broadcast_y_matmul(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2, except_split=True):
                for y_sbp in all_sbp(placement, max_dim=2, valid_split_axis=[0]):
                    _test_flow_tensor_global_x_broadcast_y_matmul(
                        test_case, placement, x_sbp, y_sbp
                    )

    @globaltest
    def test_global_broadcast_matmul_with_same_dims(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=2):
                for y_sbp in all_sbp(placement, max_dim=2):
                    _test_flow_tensor_global_broadcast_matmul_with_same_dims(
                        test_case, placement, x_sbp, y_sbp
                    )


if __name__ == "__main__":
    unittest.main()
