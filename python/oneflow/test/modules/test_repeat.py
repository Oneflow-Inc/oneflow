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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestRepeat(flow.unittest.TestCase):
    @autotest(n=10)
    def test_flow_tensor_repeat_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2)
        sizes = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        y = x.repeat(sizes)
        return y

    @autotest(n=10, auto_backward=False)
    def test_flow_tensor_repeat_bool_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2).to(torch.bool)
        sizes = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        y = x.repeat(sizes)
        return y

    @autotest(n=10)
    def test_flow_tensor_repeat_with_0dim_data(test_case):
        x = random_tensor(ndim=0)
        sizes = (random(1, 5).to(int), random(1, 5).to(int), random(1, 5).to(int))
        y = x.repeat(sizes)
        return y

    @autotest(n=5, auto_backward=False)
    def test_complicated_repeat_case(test_case):
        x = torch.ones(224, 224)
        y = torch.triu(x, diagonal=1).repeat(32, 1, 1)
        z = y.byte()
        return z

    @autotest(n=5)
    def test_flow_tensor_0size_with_random_data(test_case):
        x = random_tensor(ndim=2, dim0=3, dim1=1)
        sizes = (1, 0)
        y = x.repeat(sizes)
        return y


if __name__ == "__main__":
    unittest.main()
