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
from torch._C import dtype
from automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class Test_New_ones(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_new_ones(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=random(),dim0=random()).to(device)
        return x.new_ones((2,3,4))


if __name__ == "__main__":
    unittest.main()