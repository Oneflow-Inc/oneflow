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
import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestTensordot(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_tensordot_intdim(test_case):
        device = random_device()
        dims = random()
        dims_list = [random().to(int).value() for i in range(dims.to(int).value() + 3)]
        x = random_tensor(
            ndim=3, dim0=dims_list[0], dim1=dims_list[1], dim2=dims_list[2],
        ).to(device)
        y = random_tensor(
            ndim=3,
            dim0=dims_list[0 + dims.to(int).value()],
            dim1=dims_list[1 + dims.to(int).value()],
            dim2=dims_list[2 + dims.to(int).value()],
        ).to(device)

        z = torch.tensordot(x, y, dims=3 - dims.to(int).value())
        return z

    @autotest(check_graph=True, n=1)
    def test_tensordot_tensordim(test_case):
        device = random_device()
        x = random_tensor(4, 1, 3, 2, 5).to(device)
        y = random_tensor(4, 4, 2, 3, 5).to(device)
        z = torch.tensordot(x, y, dims=[[1, 2, 0], [2, 1, 0]])
        return z

    @autotest(check_graph=True, n=1)
    def test_tensordot_neg_tensordim(test_case):
        device = random_device()
        x = random_tensor(4, 1, 3, 2, 5).to(device)
        y = random_tensor(4, 4, 2, 3, 5).to(device)
        z = torch.tensordot(x, y, dims=[[-3, -2, -4], [-2, -3, -4]])
        return z

    @autotest(check_graph=True)
    def test_tensordot_broadcast(test_case):
        device = random_device()
        x = random_tensor(4, 1, 1, 1, 1).to(device)
        y = random_tensor(4, 2, 3, 4, 5).to(device)
        z = torch.tensordot(x, y, dims=random(high=5).to(int).value())
        return z


if __name__ == "__main__":
    unittest.main()
