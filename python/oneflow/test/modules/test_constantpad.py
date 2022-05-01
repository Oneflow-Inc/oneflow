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


from oneflow.test_utils.automated_test_util import *
from oneflow.nn.common_types import _size_2_t, _size_4_t, _size_6_t
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestConstantPad1d(flow.unittest.TestCase):
    @autotest(rtol=0.001, atol=0.001)
    def test_constantpad1d_with_random_data(test_case):
        m = torch.nn.ConstantPad1d(
            padding=random(1, 6).to(_size_2_t), value=random().to(float)
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim1=random(1, 6), dim2=random(1, 6)).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestConstantPad2d(flow.unittest.TestCase):
    @autotest(rtol=0.001, atol=0.001)
    def test_constantpad2d_with_random_data(test_case):
        m = torch.nn.ConstantPad2d(
            padding=random(1, 6).to(_size_4_t), value=random().to(float)
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=4, dim1=random(1, 6), dim2=random(1, 6), dim3=random(1, 6)
        ).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestConstantPad3d(flow.unittest.TestCase):
    @autotest(rtol=0.001, atol=0.001)
    def test_constantpad3d_with_random_data(test_case):
        m = torch.nn.ConstantPad3d(
            padding=random(1, 6).to(_size_6_t), value=random().to(float)
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=5,
            dim1=random(1, 6),
            dim2=random(1, 6),
            dim3=random(1, 6),
            dim4=random(1, 6),
        ).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestFunctionalConstantPad2d(flow.unittest.TestCase):
    @autotest(n=20, rtol=0.001, atol=0.001, check_graph=True)
    def test_functional_constantpad2d(test_case):
        device = random_device()
        padding = random(-1, 6).to(_size_4_t)
        value = random().to(float)
        x = random_tensor(
            ndim=4,
            dim0=random(1, 6),
            dim1=random(1, 6),
            dim2=random(2, 6),
            dim3=random(2, 6),
        ).to(device)
        y = torch.nn.functional.pad(x, pad=padding, mode="constant", value=value)
        return y


if __name__ == "__main__":
    unittest.main()
