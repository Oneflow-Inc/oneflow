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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

import torch as torch_original
from packaging import version


@flow.unittest.skip_unless_1n1d()
class TestRemainderModule(flow.unittest.TestCase):
    @autotest(n=1, auto_backward=True)
    def test_flow_remainder_element_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        return torch.remainder(input, other)

    @autotest(n=1, auto_backward=True)
    def test_flow_remainder_element_with_0dim_data(test_case):
        device = random_device()
        input = random_tensor(ndim=0).to(device)
        other = random_tensor(ndim=0).to(device)
        return torch.remainder(input, other)

    @autotest(n=1, auto_backward=True)
    def test_flow_remainder_broadcast_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=constant(1), dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=constant(1)).to(device)
        return torch.remainder(input, other)

    @autotest(n=1, auto_backward=True)
    def test_flow_remainder_scalar_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = 3
        return torch.remainder(input, other)

    @autotest(n=1, auto_backward=False)
    def test_flow_remainder_scalar_tensor_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = 3
        other = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        return torch.remainder(input, other)

    @autotest(n=1, auto_backward=True)
    def test_remainder_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.remainder(x, 2)
        return y

    @profile(torch.remainder)
    def profile_remainder(test_case):
        torch.remainder(torch.ones(100, 100, 100), -0.5)
        torch.remainder(torch.ones(100, 100, 100), torch.ones(100, 100))


if __name__ == "__main__":
    unittest.main()
