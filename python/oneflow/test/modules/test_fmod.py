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

import random as rd
import unittest
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestFmodModule(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_flow_fmod_element_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        return torch.fmod(input, other)

    @autotest(auto_backward=False)
    def test_flow_fmod_element_with_0dim_data(test_case):
        device = random_device()
        input = random_tensor(ndim=0).to(device)
        other = random_tensor(ndim=0).to(device)
        return torch.fmod(input, other)

    @autotest(auto_backward=False)
    def test_flow_fmod_broadcast_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=constant(1), dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=constant(1)).to(device)
        return torch.fmod(input, other)

    @autotest(auto_backward=True)
    def test_flow_fmod_scalar_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = 3
        return torch.fmod(input, other)

    @autotest(auto_backward=False)
    def test_fmod_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.fmod(x, 2)
        return y


if __name__ == "__main__":
    unittest.main()
