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
class TestAddcmul(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_addcmul(test_case):
        device = random_device()
        ndim = random(low=2).to(int).value()
        shape = [random(low=2, high=4) for i in range(ndim)]

        input = random_tensor(len(shape), *shape).to(device)
        tensor1 = random_tensor(len(shape), *shape).to(device)
        tensor2 = random_tensor(len(shape), *shape).to(device)
        value = random(3, 6).to(int)
        output = torch.addcmul(input, tensor1, tensor2, value=value)
        return output

    @autotest(check_graph=True)
    def test_tensor_addcmul(test_case):
        device = random_device()
        ndim = random(low=2).to(int).value()
        shape = [random(low=2, high=4) for i in range(ndim)]

        input = random_tensor(len(shape), *shape).to(device)
        tensor1 = random_tensor(len(shape), *shape).to(device)
        tensor2 = random_tensor(len(shape), *shape).to(device)
        value = random(3, 6).to(int)
        output = input.addcmul(tensor1, tensor2, value=value)
        return output

    @autotest(check_graph=True)
    def test_tensor_addcmul_inplace(test_case):
        device = random_device()
        ndim = random(low=2).to(int).value()
        shape = [random(low=2, high=4) for i in range(ndim)]

        input = random_tensor(len(shape), *shape).to(device)
        input = input + 1.0
        tensor1 = random_tensor(len(shape), *shape).to(device)
        tensor2 = random_tensor(len(shape), *shape).to(device)
        value = random(3, 6).to(int)
        input.addcmul_(tensor1, tensor2, value=value)
        return input

    @profile(torch.addcmul)
    def profile_addcmul(test_case):
        input = torch.ones(100, 12, 13)
        tensor1 = torch.ones(100, 12, 13)
        tensor2 = torch.ones(100, 12, 13)
        torch.addcmul(input, tensor1, tensor2, value=2)


if __name__ == "__main__":
    unittest.main()
