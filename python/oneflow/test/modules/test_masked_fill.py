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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestMaskedFill(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_masked_fill_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        input = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        mask = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        value = random().to(float)
        return input.masked_fill(mask > 0, value)

    @autotest(check_graph=True)
    def test_flow_masked_fill_broadcast_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        input = random_pytorch_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        mask = random_pytorch_tensor(ndim=2, dim0=k1, dim1=1).to(device)
        value = random().to(float)
        return input.masked_fill(mask > 0, value)

    @autotest(check_graph=True)
    def test_flow_masked_fill_int_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        input = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        mask = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        value = random().to(int)
        return input.masked_fill(mask > 0, value)


if __name__ == "__main__":
    unittest.main()
