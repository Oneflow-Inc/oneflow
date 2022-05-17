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
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestTensorSplitVec(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_tensor_split_vec(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(3, 6),
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        dim = random(-3, 3).to(int)
        z = torch.tensor_split(x, (1, 2), dim)
        return z[0]

    @autotest(n=5)
    def test_flow_tensor_split_vec_with_stride(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(3, 6),
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        dim = random(-3, 3).to(int)
        perm = [0, 1, 2, 3]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.tensor_split(y, (1, 2), dim)
        return z[0]


@flow.unittest.skip_unless_1n1d()
class TestTensorSplitInt(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_tensor_split_int(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(3, 6),
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        split = random(1, 3).to(int)
        dim = random(-3, 3).to(int)
        z = torch.tensor_split(x, split, dim)
        return z[0]


if __name__ == "__main__":
    unittest.main()
