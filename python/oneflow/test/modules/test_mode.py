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
class TestModeModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_mode_reduce_one_dim(test_case):
        device = cpu_device()
        ndim = random(low=2).to(int).value()
        reduce_dim = random(high=ndim).to(int).value()
        x = random_tensor(ndim).to(device)
        return torch.mode(x, reduce_dim)

    @autotest(n=5)
    def test_mode_reduce_one_dim_keepdim(test_case):
        device = cpu_device()
        ndim = random(low=2).to(int).value()
        reduce_dim = random(high=ndim).to(int).value()
        x = random_tensor(ndim).to(device)
        return torch.mode(x, reduce_dim, True)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_mode_0size(test_case):
        device = cpu_device()
        x = random_tensor(ndim=3, dim1=0, requires_grad=False).to(device)
        return torch.mode(x)

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_mode_reduce_one_dim_0size(test_case):
        device = cpu_device()
        x = random_tensor(ndim=3, dim1=0, requires_grad=False).to(device)
        return torch.mode(x, 0)


if __name__ == "__main__":
    unittest.main()
