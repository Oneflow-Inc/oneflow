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
from random import shuffle

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestAsStrided(flow.unittest.TestCase):
    @autotest(n=10)
    def test_flow_AsStrided(test_case):
        device = random_device()
        ndim = np.random.randint(3, 6)
        dim0 = np.random.randint(2, 4)
        dim1 = np.random.randint(2, 4)
        dim2 = np.random.randint(2, 4)
        dim3 = np.random.randint(2, 4)
        dim4 = np.random.randint(2, 4)
        if ndim == 3:
            x = random_tensor(3, dim0, dim1, dim2)
        elif ndim == 4:
            x = random_tensor(4, dim0, dim1, dim2, dim3)
        elif ndim == 5:
            x = random_tensor(5, dim0, dim1, dim2, dim3, dim4)
        x = x.to(device)
        storage_offset = random(0, 3).to(int)
        z = torch.as_strided(x, (2, 2, 3), (1, 1, 2), storage_offset)
        return z

    # TODO:(zhaoluyang) some bug in as_strided backward to be fixed, related to the view mechanism.
    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_flow_as_strided_with_stride(test_case):
        device = random_device()
        dim0 = np.random.randint(2, 4)
        dim1 = np.random.randint(2, 4)
        dim2 = np.random.randint(2, 4)
        dim3 = np.random.randint(2, 4)
        x = random_tensor(4, dim0, dim1, dim2, dim3)
        x = x.to(device)
        storage_offset = random(0, 3).to(int)
        perm = [0, 1, 2, 3]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.as_strided(y, (2, 2, 3), (1, 1, 2), storage_offset)
        return z

    @autotest(n=10, auto_backward=False)
    def test_flow_as_strided_bool(test_case):
        device = random_device()
        ndim = np.random.randint(3, 6)
        dim0 = np.random.randint(2, 4)
        dim1 = np.random.randint(2, 4)
        dim2 = np.random.randint(2, 4)
        dim3 = np.random.randint(2, 4)
        dim4 = np.random.randint(2, 4)
        if ndim == 3:
            x = random_tensor(3, dim0, dim1, dim2)
        elif ndim == 4:
            x = random_tensor(4, dim0, dim1, dim2, dim3)
        elif ndim == 5:
            x = random_tensor(5, dim0, dim1, dim2, dim3, dim4)
        x = x.to(device)
        x = x.to(torch.bool)
        storage_offset = random(0, 3).to(int)
        z = torch.as_strided(x, (2, 2, 3), (1, 1, 2), storage_offset)
        return z


if __name__ == "__main__":
    unittest.main()
