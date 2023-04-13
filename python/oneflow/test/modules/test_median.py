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


@flow.unittest.skip_unless_1n1d()
class TestMedianModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_median_reduce_all_dim(test_case):
        device = random_device()
        ndim = random(1, 4).to(int).value()
        x = random_tensor(ndim=ndim, dim0=random(1, 4)).to(device)
        return torch.median(x)

    @autotest(n=5)
    def test_median_reduce_one_dim(test_case):
        device = random_device()
        ndim = random(low=2).to(int).value()
        reduce_dim = random(high=ndim).to(int).value()
        x = random_tensor(ndim).to(device)
        return torch.median(x, reduce_dim)

    @autotest(n=5)
    def test_median_reduce_one_dim_keepdim(test_case):
        device = random_device()
        ndim = random(low=2).to(int).value()
        reduce_dim = random(high=ndim).to(int).value()
        x = random_tensor(ndim).to(device)
        return torch.median(x, reduce_dim, True)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_median_0size(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim1=0, requires_grad=False).to(device)
        return torch.median(x)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_median_reduce_one_dim_0size(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim1=0, requires_grad=False).to(device)
        return torch.median(x, 0)

    @autotest(n=5, auto_backward=False)
    def test_median_return_type(test_case):
        x = random_tensor(3, 4)
        result = x.median(1)
        return result.values, result.indices


if __name__ == "__main__":
    unittest.main()
