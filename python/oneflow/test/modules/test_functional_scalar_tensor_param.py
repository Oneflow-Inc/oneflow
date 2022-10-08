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
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestFunctionalWithScalarTensorParam(flow.unittest.TestCase):
    # NOTE: graph mode not support dynamic scalar tensor parameter
    @autotest(n=2, auto_backward=False, check_graph=False)
    def test_scalar_tensor_transfer_to_scalar(test_case):
        device = random_device()
        min = torch.tensor(0.0)
        max = torch.tensor(0.5)
        x = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        return x.clamp(min=min, max=max)

    @autotest(n=2, auto_backward=False, check_graph=False)
    def test_scalar_tensor_transfer_to_double(test_case):
        device = random_device()
        threshold = torch.tensor(0.5).to(device)
        x = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        return torch.nn.functional.threshold(x, threshold=threshold, value=0.5)

    @autotest(n=2, auto_backward=False, check_graph=False)
    def test_scalar_tensor_transfer_to_int(test_case):
        device = random_device()
        start_dim = torch.tensor(1).to(device)
        end_dim = torch.tensor(3).to(device)
        x = random_tensor(4, *(2, 3, 4, 5)).to(device)
        return x.flatten(start_dim=start_dim, end_dim=end_dim)


if __name__ == "__main__":
    unittest.main()
