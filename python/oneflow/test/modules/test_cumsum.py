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
class TestCumsum(flow.unittest.TestCase):
    @autotest(n=30, check_graph=True)
    def test_cumsum(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        dim = random(0, x.ndim.pytorch).to(int)
        z = torch.cumsum(x, dim)
        return z


if __name__ == "__main__":
    unittest.main()
