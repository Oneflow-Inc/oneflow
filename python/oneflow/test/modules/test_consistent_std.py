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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(n=3, auto_backward=False, rtol=0.01, atol=0.01, check_graph=False)
def _test_consistent_std_flow_with_random_data(test_case, placement, sbp):
    dim = random(low=0, high=4).to(int)
    x = random_pytorch_tensor(
        ndim=4,
        dim0=random(2, 4) * 8,
        dim1=random(2, 4) * 8,
        dim2=random(2, 4) * 8,
        dim3=random(2, 4) * 8,
    ).to_consistent(placement, sbp)
    z = torch.std(x, dim=dim, unbiased=random().to(bool), keepdim=random().to(bool),)
    return z


@autotest(n=3, auto_backward=False, rtol=0.01, atol=0.01, check_graph=False)
def _test_consistent_std_tensor_with_random_data(test_case, placement, sbp):
    dim = random(low=0, high=4).to(int)
    x = random_pytorch_tensor(
        ndim=4,
        dim0=random(2, 4) * 8,
        dim1=random(2, 4) * 8,
        dim2=random(2, 4) * 8,
        dim3=random(2, 4) * 8,
    ).to_consistent(placement, sbp)
    z = x.std(dim=dim, keepdim=random().to(bool),)
    return z


class TestConsistentStd(flow.unittest.TestCase):
    @consistent
    def test_consistent_std_flow_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_consistent_std_flow_with_random_data(test_case, placement, sbp)

    @consistent
    def test_consistent_std_tensor_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_consistent_std_tensor_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
