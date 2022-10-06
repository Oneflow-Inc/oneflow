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
class TestMovedim(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_movedim_with_vector(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
            dim4=random(3, 6),
        ).to(device)
        z = torch.movedim(x, (0, 1), (2, 3))
        return z

    @autotest(n=10)
    def test_flow_movedim_with_stride(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
            dim4=random(3, 6),
        ).to(device)
        perm = [0, 1, 2, 3]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.movedim(y, (0, 1), (2, 3))
        return z

    @autotest(check_graph=True)
    def test_flow_movedim_with_int(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4,
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
            dim4=random(3, 6),
        ).to(device)
        z = torch.movedim(x, 0, 3)
        return z


if __name__ == "__main__":
    unittest.main()
