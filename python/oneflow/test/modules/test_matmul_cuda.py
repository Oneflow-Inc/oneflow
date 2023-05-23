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
import oneflow as flow
import oneflow.unittest
import torch as torch_original

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    @autotest(check_graph=True, rtol=1e-2, atol=1e-4)
    def test_flow_tensor_matmul_with_random_data_allow_tf32(test_case):
        flow.backends.cuda.matmul.allow_tf32 = True
        torch_original.backends.cuda.matmul.allow_tf32 = True
        device = random_device()
        k = random(1, 6)
        x = random_tensor(ndim=2, dim1=k).to(device)
        y = random_tensor(ndim=2, dim0=k).to(device)
        return x.matmul(y)


if __name__ == "__main__":
    unittest.main()
