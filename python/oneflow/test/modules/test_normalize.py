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

class TestNormalize(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_normalize(test_case):
        device = random_device()
        p = random(1, 4).to(float)
        dim = random(-3, 2).to(int)
        eps = random(1e-12, 1)
        x = random_pytorch_tensor(
            ndim=3,
            dim1=random(3, 6),
            dim2=random(3, 6),
            dim3=random(3, 6),
        ).to(device)
        z = torch.nn.functional.normalize(x, p, dim, eps)
        return z

if __name__ == "__main__":
    unittest.main()
