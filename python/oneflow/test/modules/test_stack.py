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
class TestStackModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_stack_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(device)
        y = random_pytorch_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(device)
        out = torch.stack((x, y), dim=random(low=1, high=4).to(int))
        return out


if __name__ == "__main__":
    unittest.main()
