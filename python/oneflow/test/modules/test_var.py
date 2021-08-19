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
class TestVar(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_var_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(2,3,4,5,requires_grad=False).to(device)
        y = torch.var(x)
        return y

    
    # @autotest()
    # def test_var_with_random_data_dim(test_case):
    #     device = random_device()
    #     x = random_pytorch_tensor(ndim=4, dim0=4).to(device)
    #     k = random(1,4).to(int)
    #     y = torch.var(x, dim=k)
    #     return y
    

if __name__ == "__main__":
    unittest.main()
