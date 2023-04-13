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
class TestAddcdiv(flow.unittest.TestCase):
    @autotest(n=5)
    def test_addcdiv(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        tensor1 = random_tensor(ndim, *shape).to(device)
        tensor2 = random_tensor(ndim, *shape).to(device)
        value = random(2, 4).to(int)
        output = torch.addcdiv(input, tensor1, tensor2, value=value)
        return output

    @autotest(n=5)
    def test_tensor_addcdiv(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        tensor1 = random_tensor(ndim, *shape).to(device)
        tensor2 = random_tensor(ndim, *shape).to(device)
        value = random(2, 4).to(int)
        output = input.addcdiv(tensor1, tensor2, value=value)
        return output

    @autotest(n=5)
    def test_tensor_addcdiv_inplace(test_case):
        device = random_device()
        ndim = random(2, 4).to(int).value()
        shape = [random(2, 4) for i in range(ndim)]
        input = random_tensor(ndim, *shape).to(device)
        input = input + 1.0
        tensor1 = random_tensor(ndim, *shape).to(device)
        tensor2 = random_tensor(ndim, *shape).to(device)
        value = random(2, 4).to(int)
        input.addcdiv_(tensor1, tensor2, value=value)
        return input

    @profile(torch.addcdiv)
    def profile_addcdiv(test_case):
        t = torch.ones(1, 3)
        t1 = torch.ones(3, 1)
        t2 = torch.ones(1, 3)
        torch.addcdiv(t, t1, t2, value=0.1)


if __name__ == "__main__":
    unittest.main()
