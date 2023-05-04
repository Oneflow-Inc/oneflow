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
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestDot(flow.unittest.TestCase):
    @autotest(n=5)
    def test_dot(test_case):
        device = random_device()
        k = random(10, 100)
        x = random_tensor(ndim=1, dim0=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        z = torch.dot(x, y)
        return z

    @autotest(n=5, check_graph=False)
    def test_dot_with_random_int_data(test_case):
        k = np.random.randint(0, 100)
        x = np.random.randint(low=0, high=100, size=k)
        y = np.random.randint(low=0, high=100, size=k)
        torch_x = torch.from_numpy(x).to(torch.int)
        torch_y = torch.from_numpy(y).to(torch.int)
        torch_output_numpy = torch.dot(torch_x, torch_y).numpy()
        flow_x = flow.tensor(x).to(flow.int)
        flow_y = flow.tensor(y).to(flow.int)
        flow_output_numpy = flow.dot(flow_x, flow_y).numpy()
        test_case.assertTrue(
            np.allclose(flow_output_numpy, torch_output_numpy, 1e-05, 1e-05)
        )

    @profile(torch.dot)
    def profile_dot(test_case):
        input1 = torch.ones(10000)
        input2 = torch.ones(10000)
        torch.dot(input1, input2)


if __name__ == "__main__":
    unittest.main()
