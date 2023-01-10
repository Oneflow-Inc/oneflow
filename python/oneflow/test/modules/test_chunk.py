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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestChunk(flow.unittest.TestCase):
    @autotest(n=5, check_graph=True)
    def test_flow_chunk_list_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        y = torch.chunk(x, chunks=random(low=1, high=5).to(int), dim=dim)
        z = torch.cat(y, dim=dim)
        return z

    @autotest(n=10)
    def test_flow_chunk_list_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        permute_list = [0, 1, 2, 3]
        shuffle(permute_list)
        y = x.permute(permute_list)
        z = torch.chunk(y, chunks=random(low=1, high=5).to(int), dim=dim)
        return torch.cat(z, dim=dim)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_chunk_list_with_stride(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        perm = [0, 1, 2, 3]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.chunk(y, chunks=random(low=1, high=5).to(int), dim=dim)
        return torch.cat(z, dim=dim)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_chunk_list_bool_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device, torch.bool)
        y = torch.chunk(x, chunks=random(low=1, high=5).to(int), dim=dim)
        z = torch.cat(y, dim=dim)
        return z

    @autotest(n=5, check_graph=True)
    def test_flow_chunk_list_with_random_data_negative_dim(test_case):
        device = random_device()
        dim = random(1, 3).to(int)
        x = random_tensor(
            ndim=4,
            dim0=random(low=4, high=8).to(int),
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        y = torch.chunk(x, chunks=4, dim=-1)
        z = torch.cat(y, dim=-1)
        return z

    @profile(torch.chunk)
    def profile_chunk(test_case):
        torch.chunk(torch.ones(16), 4)
        torch.chunk(torch.ones(100000), 5)
        torch.chunk(torch.ones(100, 100), 5, dim=1)


if __name__ == "__main__":
    unittest.main()
