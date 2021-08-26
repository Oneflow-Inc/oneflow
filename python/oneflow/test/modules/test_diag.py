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
from automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class Test_Diag_Module(flow.unittest.TestCase):
    @autotest
    def test_diag_one_dim(test_case):
        device = random_device()
        k = random()
        x = random_pytorch_tensor(ndim=1, dim0=k).to(device)
        return torch.diag(x)

    @autotest
    def test_diag_other_dim(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim0=random(), dim1=random()).to(device)
        return torch.diag(x)

    @autotest
    def test_tensor_diag_one_dim(test_case):
        device = random_device()
        k = random()
        x = random_pytorch_tensor(ndim=1, dim0=k).to(device)
        return x.diag()

    @autotest
    def test_tensor_diag_other_dim(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim0=random(), dim1=random()).to(device)
        return x.diag()

if __name__ == "__main__":
    unittest.main()