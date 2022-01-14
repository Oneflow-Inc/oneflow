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
from oneflow.test_utils.automated_test_util.generators import random
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


class TestVar(flow.unittest.TestCase):
    def test_flow_var_all_dim_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.var(x)
        return y

    @autotest(check_graph=False)
    def test_flow_var_one_dim_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4).to(device)
        y = torch.var(
            x,
            dim=random(low=0, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    # TODO(): 'var backward' is composed of several other ops,
    # reducemean doesn't support 0-shape for now
    @autotest(auto_backward=False, check_graph=False)
    def test_flow_var_0_size_data_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 3, 0, 4).to(device)
        y = torch.var(
            x,
            dim=random(low=0, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y


if __name__ == "__main__":
    unittest.main()
