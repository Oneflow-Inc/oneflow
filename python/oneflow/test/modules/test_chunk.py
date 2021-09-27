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

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestChunk(flow.unittest.TestCase):
    @autotest()
    def test_flow_chunk_list_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_pytorch_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        y = torch.chunk(x, chunks=random(low=1, high=5).to(int), dim=dim)
        z = torch.cat(y, dim=dim)
        return z


if __name__ == "__main__":
    unittest.main()
