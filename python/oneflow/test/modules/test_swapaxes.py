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
from random import shuffle

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestSwapaxes(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_swapaxes_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        y = torch.swapaxes(x, random(0, 2).to(int), random(0, 2).to(int))
        return y

    @autotest(n=10)
    def test_swapaxes_flow_with_stride(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        perm = [0, 1, 2]
        shuffle(perm)
        y = x.permute(perm)
        z = torch.swapaxes(y, random(0, 2).to(int), random(0, 2).to(int))
        return z


if __name__ == "__main__":
    unittest.main()
