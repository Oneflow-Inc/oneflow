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
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestFrac(flow.unittest.TestCase):
    @autotest(n=5)
    def test_frac(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        output = torch.frac(input)
        return output

    @autotest(n=5)
    def test_tensor_frac(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        output = input.frac()
        return output

    @autotest(n=5)
    def test_tensor_frac_inplace(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        input = input + 1.0
        input.frac_()
        return input


if __name__ == "__main__":
    unittest.main()
