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
class TestLinalgVectorNorm2D(flow.unittest.TestCase):
    @autotest(n=2, auto_backward=False, check_graph=True, rtol=0.5, atol=0.5)
    def test_sqrt_sum_with_cpu_random_data(test_case):
        device = cpu_device()
        x = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5, requires_grad=False).to(
            device
        )
        y = torch.linalg.norm(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=2, auto_backward=False, check_graph=True)
    def test_sqrt_sum_with_cuda_random_data(test_case):
        device = gpu_device()
        x = random_tensor(ndim=4, dim1=10, dim2=10, dim3=10, requires_grad=False).to(
            device
        )
        y = torch.linalg.norm(x)
        return y

    @autotest(n=2, auto_backward=False, check_graph=True, rtol=0.5, atol=0.5)
    def test_scalar_print_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5, requires_grad=False).to(
            device
        )
        y = torch.linalg.norm(x)
        return y


if __name__ == "__main__":
    unittest.main()
