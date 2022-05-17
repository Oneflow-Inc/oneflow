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
from collections import OrderedDict

from random import shuffle
import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
import oneflow.unittest
import oneflow as flow


@flow.unittest.skip_unless_1n1d()
class TestContiguous(flow.unittest.TestCase):
    @autotest(n=10, check_graph=True)
    def test_transpose_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        z = y.contiguous()
        return y

    @autotest(n=10, check_graph=True)
    def test_permute2d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 2
        permute_list = [0, 1]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim, dim0=random(1, 32).to(int), dim1=random(1, 59).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z

    @autotest(n=10, check_graph=True)
    def test_permute3d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 3
        permute_list = [0, 1, 2]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim,
            dim0=random(1, 7).to(int),
            dim1=random(1, 15).to(int),
            dim2=random(1, 9).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z

    @autotest(n=10, check_graph=True)
    def test_permute4d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 4
        permute_list = [0, 1, 2, 3]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim,
            dim0=random(1, 7).to(int),
            dim1=random(1, 15).to(int),
            dim2=random(1, 9).to(int),
            dim3=random(1, 19).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z


if __name__ == "__main__":
    unittest.main()
