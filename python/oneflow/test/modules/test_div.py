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

import numpy as np
from automated_test_util import *
import oneflow as flow
import oneflow.unittest




@flow.unittest.skip_unless_1n1d()
class TestDiv(flow.unittest.TestCase):
    @autotest()
    def test_div_with_random_data_Number(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=1).to(device)
        k = random(1,4).to(float)
        y = torch.div(x, k)
        return y

    
    @autotest(auto_backward=False)
    def test_div_with_0shape(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=2, dim1=1, dim2=0, dim3=3).to(device)
        y = random_pytorch_tensor(ndim=4, dim0=2, dim1=1, dim2=0, dim3=3).to(device)
        z = torch.div(x,y)
        return z
    
    @autotest(auto_backward=False)
    def test_div_with_diff_shape(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=3).to(device)
        y = random_pytorch_tensor(ndim=4).to(device)
        z = torch.div(x, y)
        return z

    @autotest()
    def test_tensor_div_with_random_data_Number(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=1).to(device)
        k = random(1,4).to(float)
        y = x.div(k)
        return y

    
    @autotest(auto_backward=False)
    def test_tensor_div_with_0shape(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=2, dim1=1, dim2=0, dim3=3).to(device)
        y = random_pytorch_tensor(ndim=4, dim0=2, dim1=1, dim2=0, dim3=3).to(device)
        z = x.div(y)
        return z
    
    @autotest(auto_backward=False)
    def test_tensor_div_with_diff_shape(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=3).to(device)
        y = random_pytorch_tensor(ndim=4).to(device)
        z = x.div(y)
        return z


if __name__ == "__main__":
    unittest.main()
