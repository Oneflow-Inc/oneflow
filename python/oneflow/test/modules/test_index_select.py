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


import oneflow as flow
import oneflow.unittest

import unittest


from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestIndexSelect(flow.unittest.TestCase):
    @autotest()
    def test_index_select_by_random(test_case):
        device = random_device()

        # test 4 dimensions tensor
        dim = random(0, 4).to(int)

        tensor_dim = []
        for i in range(0, 4):
            tensor_dim.append(random(2, 6).to(int).value())

        index = random_tensor(
            ndim=1,
            dim0=random(1, 10).to(int),
            low=0,
            high=tensor_dim[dim.value()],
            dtype=int,
        ).to(device)

        x = random_tensor(
            ndim=4,
            dim0=tensor_dim[0],
            dim1=tensor_dim[1],
            dim2=tensor_dim[2],
            dim3=tensor_dim[3],
        ).to(device)

        y = torch.index_select(x, dim, index)

        return y

    @autotest(auto_backward=False)
    def test_index_select_bool_by_random(test_case):
        device = random_device()

        # test 4 dimensions tensor
        dim = random(0, 4).to(int)

        tensor_dim = []
        for i in range(0, 4):
            tensor_dim.append(random(2, 6).to(int).value())

        index = random_tensor(
            ndim=1,
            dim0=random(1, 10).to(int),
            low=0,
            high=tensor_dim[dim.value()],
            dtype=int,
        ).to(device)

        x = random_tensor(
            ndim=4,
            dim0=tensor_dim[0],
            dim1=tensor_dim[1],
            dim2=tensor_dim[2],
            dim3=tensor_dim[3],
        ).to(device=device, dtype=torch.bool)

        y = torch.index_select(x, dim, index)

        return y


if __name__ == "__main__":
    unittest.main()
