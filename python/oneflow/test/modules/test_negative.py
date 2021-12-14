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
class TestNegativeModule(flow.unittest.TestCase):
    @autotest(auto_backward=False, check_graph=False)
    def test_ne_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 3, 0, 5).to(device)
        y1 = torch.negative(x)
        y2 = torch.neg(x)
        y3 = torch.negative_(x)
        y4 = torch.neg_(x)
        y5 = -x
        return (y1, y2, y3, y4, y5)

    @autotest(check_graph=False)
    def test_tensor_negative_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        return x.negative()

    @autotest(check_graph=False)
    def test_negative_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = torch.negative(x)
        return z

    @autotest(check_graph=False)
    def test_neg_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = torch.neg(x)
        return z

    @autotest(check_graph=False)
    def test_neg_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = x.neg()
        return z

    @autotest(check_graph=False)
    def test_tensor_inplace_negative_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        return x.negative_()

    @autotest(check_graph=False)
    def test_inplace_negative_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = torch.negative_(x)
        return z

    @autotest(check_graph=False)
    def test_inplace_neg_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = torch.neg_(x)
        return z

    @autotest(check_graph=False)
    def test_inplace_neg_with_random_data(test_case):
        x = random_pytorch_tensor().to(random_device())
        z = x.neg_()
        return z


if __name__ == "__main__":
    unittest.main()
