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
from random import shuffle

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.automated_test_util import util
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestSelect(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_select(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(3, 6),
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        dim = random(-4, 3).to(int)
        index = random(0, 2).to(int)
        z = torch.select(x, dim, index)
        return z

    # TODO:(zhaoluyang) some bug in as_strided backward to be fixed
    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_flow_select_with_stride(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(3, 6),
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        dim = random(-4, 3).to(int)
        index = random(0, 2).to(int)
        perm = [0, 1, 2, 3]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.select(y, dim, index)
        return z

    @autotest(check_graph=True)
    def test_flow_select_1dim(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random(3, 6),).to(device)
        index = random(0, 2).to(int)
        z = torch.select(x, 0, index)
        return z


if __name__ == "__main__":
    unittest.main()
