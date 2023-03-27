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
from oneflow.test_utils.automated_test_util.generators import random
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_flow_global_var_all_dim_with_random_data(test_case, placement, sbp):
    x = random_tensor(
        ndim=2, dim0=random(1, 3).to(int) * 8, dim1=random(1, 3).to(int) * 8,
    ).to_global(placement, sbp)
    y = torch.var(x)
    return y


@autotest(n=1, check_graph=True)
def _test_flow_global_var_one_dim_with_random_data(test_case, placement, sbp):
    x = random_tensor(
        ndim=2, dim0=random(1, 3).to(int) * 8, dim1=random(1, 3).to(int) * 8,
    ).to_global(placement, sbp)
    y = torch.var(
        x,
        dim=random(low=0, high=2).to(int),
        unbiased=random().to(bool),
        keepdim=random().to(bool),
    )
    return y


@autotest(n=1, auto_backward=True, check_graph=True)
def _test_flow_var_0_size_data_with_random_data(test_case, placement, sbp):
    x = random_tensor(3, 8, 0, 8).to_global(placement, sbp)
    y = torch.var(
        x,
        dim=random(low=0, high=3).to(int),
        unbiased=random().to(bool),
        keepdim=random().to(bool),
    )
    return y


class TestVar(flow.unittest.TestCase):
    @globaltest
    def test_flow_global_var_all_dim_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_global_var_all_dim_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_global_var_one_dim_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_flow_global_var_one_dim_with_random_data(
                    test_case, placement, sbp
                )

    @globaltest
    def test_flow_var_0_size_data_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2, valid_split_axis=[0]):
                _test_flow_var_0_size_data_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
