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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestUnfoldTensor(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=True, check_graph=True)
    def test_unfold_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(3, 3, 4, 5).to(device)
        dimension = random(0, 2).to(int).value()
        size = random(1, 3).to(int).value()
        step = random(1, 3).to(int).value()
        y = x.unfold(dimension, size, step)
        return y

    @autotest(n=5)
    def test_unfold_tensor_with_stride(test_case):
        device = random_device()
        x = random_tensor(3, 3, 4, 5).to(device)
        perm = [0, 1, 2]
        shuffle(perm)
        y = x.permute(perm)
        dimension = random(0, 2).to(int).value()
        size = random(1, 3).to(int).value()
        step = random(1, 3).to(int).value()
        z = y.unfold(dimension, size, step)
        return z

    @autotest(n=10, auto_backward=True, check_graph=True)
    def test_unfold_tensor_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        dimension = random(0, 2).to(int).value()
        size = random(1, 3).to(int).value()
        step = random(1, 3).to(int).value()
        y = x.unfold(dimension, size, step)
        return y


if __name__ == "__main__":
    unittest.main()
