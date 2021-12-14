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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestReciprocalModule(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_reciprocal_list_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        y = torch.reciprocal(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_reciprocal_list_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        y = x.reciprocal()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_reciprocal_list_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.reciprocal_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_reciprocal_list_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        x = x_0 + 1
        id_x = id(x)
        x.reciprocal_()
        test_case.assertTrue(id_x == id(x))
        return x


if __name__ == "__main__":
    unittest.main()
