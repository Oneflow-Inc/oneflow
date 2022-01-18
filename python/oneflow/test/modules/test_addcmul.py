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


class TestAddcmul(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_Addcmul(test_case):
        device = random_device()
        input = random_pytorch_tensor(2,3,4).to(device)
        tensor1 = random_pytorch_tensor(2,3,4).to(device)
        tensor2 = random_pytorch_tensor(2,3,4).to(device)
        value = random(3, 6).to(int)
        z = torch.addcmul(input, tensor1, tensor2, value=value)
        return z
    
    @autotest(check_graph=False)
    def test_TensorAddcmul(test_case):
        device = random_device()
        input = random_pytorch_tensor(2,3,4).to(device)
        tensor1 = random_pytorch_tensor(2,3,4).to(device)
        tensor2 = random_pytorch_tensor(2,3,4).to(device)
        value = random(3, 6).to(int)
        z = input.addcmul(tensor1, tensor2, value=value)
        return z
    
    @autotest(check_graph=False)
    def test_TensorAddcmulInplace(test_case):
        device = random_device()
        input = random_pytorch_tensor(2,3,4).to(device)
        input1 = input + 1
        tensor1 = random_pytorch_tensor(2,3,4).to(device)
        tensor2 = random_pytorch_tensor(2,3,4).to(device)
        value = random(3, 6).to(int)
        input1.addcmul_(tensor1, tensor2, value=value)
        return input1

if __name__ == "__main__":
    unittest.main()
