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
    def test_tensordot(test_case):
        device = random_device()
        dims = random()
        dims_list = [random(high=4).to(int).value() for i in range(dims.to(int).value() + 4)]
        x = random_tensor(
            ndim=4,
            dim0=dims_list[0],
            dim1=dims_list[1],
            dim2=dims_list[2],
            dim3=dims_list[3],
        ).to(device)
        y = random_tensor(
            ndim=4,
            dim0=dims_list[0 + dims.to(int).value()],
            dim1=dims_list[1 + dims.to(int).value()],
            dim2=dims_list[2 + dims.to(int).value()],
            dim3=dims_list[3 + dims.to(int).value()],
        ).to(device)

        z = torch.tensordot(x, y, dims=4 - dims.to(int).value())
        return z


if __name__ == "__main__":
    unittest.main()
